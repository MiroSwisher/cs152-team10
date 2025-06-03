# bot.py
import discord
from discord.ext import commands
import os
import json
import logging
import re
import requests
from report import Report
import pdb
from classifier import load_model, predict_severity

# Set up logging to the console
logger = logging.getLogger('discord')
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')
handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s'))
logger.addHandler(handler)

# There should be a file called 'tokens.json' inside the same folder as this file
token_path = 'tokens.json'
if not os.path.isfile(token_path):
    raise Exception(f"{token_path} not found!")
with open(token_path) as f:
    # If you get an error here, it means your token is formatted incorrectly. Did you put it in quotes?
    tokens = json.load(f)
    discord_token = tokens['discord']


class ModBot(discord.Client):
    def __init__(self): 
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix='.', intents=intents)
        self.group_num = None
        self.mod_channels = {}  # Map from guild to the mod channel id for that guild
        self.reports = {}       # Map from user IDs to the state of their report
        # Storage for completed reports and moderation actions
        self.report_store = {}  # report_id -> metadata dict
        self.next_report_id = 1  # incremental report ID
        # Enforcement lists
        self.shadow_blocked = set()  # user IDs whose messages are hidden
        self.blocked_users = set()   # user IDs who are fully blocked
        # Tracker for offenders' prior violation counts
        self.violation_history = {}  # offender_id -> violation count
        # Optional: track reporters for false-reporting flags
        self.flagged_reporters = set()
        # Load hate speech classifier and define severity labels
        self.vectorizer, self.classifier = load_model()
        self.severity_labels = {
            0: "Non-Hateful",
            1: "Mild Hate",
            2: "Moderate Hate",
            3: "Severe Hate",
            4: "Extremist Hate"
        }

    async def on_ready(self):
        print(f'{self.user.name} has connected to Discord! It is these guilds:')
        for guild in self.guilds:
            print(f' - {guild.name}')
        print('Press Ctrl-C to quit.')

        # Parse the group number out of the bot's name
        match = re.search('[gG]roup (\d+) [bB]ot', self.user.name)
        if match:
            self.group_num = match.group(1)
        else:
            raise Exception("Group number not found in bot's name. Name format should be \"Group # Bot\".")

        # Find the mod channel in each guild that this bot should report to
        for guild in self.guilds:
            for channel in guild.text_channels:
                if channel.name == f'group-{self.group_num}-mod':
                    self.mod_channels[guild.id] = channel
        

    async def on_message(self, message):
        """Route messages: enforce shadow/block, handle main channel, mod commands, or DMs."""
        # Ignore bot's own messages
        if message.author.id == self.user.id:
            return

        if message.guild:
            # Enforce shadow-block: delete messages silently
            if message.channel.name == f'group-{self.group_num}':
                if message.author.id in self.shadow_blocked:
                    await message.delete()
                    return
                if message.author.id in self.blocked_users:
                    await message.delete()
                    try:
                        await message.author.send("You are blocked from sending messages here.")
                    except discord.Forbidden:
                        pass
                    return
                # Handle normal main channel messages
                await self.handle_channel_message(message)
                return
            # Handle moderator commands in mod channel
            if message.channel.name == f'group-{self.group_num}-mod':
                await self.handle_mod_message(message)
                return
            # Other guild channels: ignore
            return

        # Direct messages (user reporting flow)
        await self.handle_dm(message)

    async def handle_dm(self, message):
        # Handle a help message
        if message.content.lower() == Report.HELP_KEYWORD:
            await message.channel.send(
                "Click the 🚩 reaction on any message in the main channel to start a report."
            )
            return

        author_id = message.author.id
        # Only respond if user has an active report flow
        if author_id not in self.reports:
            return

        report = self.reports[author_id]
        responses = await report.handle_message(message)
        for r in responses:
            await message.channel.send(r)

        # If the report is complete or cancelled, remove it from our map
        if report.report_complete():
            self.reports.pop(author_id)

    async def handle_channel_message(self, message):
        # Only handle messages sent in the "group-#" channel
        if not message.channel.name == f'group-{self.group_num}':
            return

        # Automated detection and enforcement
        severity = self.eval_text(message.content)
        await self.take_action(severity, message)

        # Forward the message to the mod channel with severity info
        mod_channel = self.mod_channels[message.guild.id]
        await mod_channel.send(f'Forwarded message:\n{message.author.name}: "{message.content}"')
        await mod_channel.send(self.code_format(severity))

    def eval_text(self, message):
        """
        Run the hate speech classifier and return a severity level (0-4).
        """
        return predict_severity(message, self.vectorizer, self.classifier)

    def code_format(self, text):
        """
        Format the classifier output for the mod channel.
        """
        severity = text
        label = self.severity_labels.get(severity, "Unknown")
        return f"Automated detection severity: {severity} ({label})"

    async def handle_mod_message(self, message):
        """Handle moderator commands in the mod channel."""
        content = message.content.strip()
        # Use '.' prefix for mod commands
        if not content.startswith('.'):
            return
        parts = content[1:].split()
        if not parts:
            return
        cmd = parts[0].lower()
        args = parts[1:]
        # Helper to send command usage
        async def send_usage():
            usage = (
                "Moderator commands:\n"
                ".list_reports - List all pending reports\n"
                ".show <report_id> - Show details of a pending report\n"
                ".dismiss <report_id> - No action\n"
                ".remove <report_id> - Delete the offending message\n"
                ".warn <report_id> [reason] - Send a warning to the user\n"
                ".shadow_block <report_id> - Hide user's messages\n"
                ".unshadow <user_id> - Remove shadow block from user\n"
                ".block <report_id> - Block user from messaging\n"
                ".unblock <user_id> - Remove block from user\n"
                ".escalate <report_id> - Escalate to higher admin\n"
                ".help - Show this help message\n"
            )
            await message.channel.send(usage)
        if cmd == 'help':
            await send_usage()
            return
        # List all open report IDs with summaries
        if cmd == 'list_reports':
            if not self.report_store:
                await message.channel.send("No pending reports.")
            else:
                lines = ["Reports (including resolved):"]
                for rid, data in self.report_store.items():
                    status = data.get('status', 'pending')
                    lines.append(
                        f"#{rid}: {data['abuse_type'].title()} (status: {status}), reporter <@{data['reporter_id']}>, offender <@{data['offender_id']}>"
                    )
                await message.channel.send("\n".join(lines))
            return
        # Show detailed embed for a specific report
        if cmd == 'show':
            if not args:
                await message.channel.send("Please specify a report ID. Use .help for commands.")
                return
            try:
                rid = int(args[0])
            except ValueError:
                await message.channel.send("Report ID must be an integer.")
                return
            if rid not in self.report_store:
                await message.channel.send(f"Report ID {rid} not found.")
                return
            # Reconstruct and send an embed using stored metadata
            data = self.report_store[rid]
            embed = discord.Embed(title=f"Report #{rid} Details", color=discord.Color.blue())
            embed.set_footer(text=f"Report ID: {rid}")
            embed.add_field(name="Reporter", value=f"<@{data['reporter_id']}>", inline=True)
            embed.add_field(name="Category", value=data['abuse_type'], inline=True)
            if data.get('subtype'):
                embed.add_field(name="Subtype", value=data['subtype'], inline=True)
            if 'filter_opt_in' in data and data['filter_opt_in'] is not None:
                embed.add_field(name="Filter Opt-In", value=str(data['filter_opt_in']), inline=True)
            if 'block_opt_in' in data and data['block_opt_in'] is not None:
                embed.add_field(name="Block Opt-In", value=str(data['block_opt_in']), inline=True)
            # violation history
            hist = self.violation_history.get(data['offender_id'], 0)
            embed.add_field(name="Culprit history of violations", value=str(hist), inline=False)
            embed.add_field(name="Message Link", value=data['report_link'], inline=False)
            embed.add_field(name="Offending User", value=f"<@{data['offender_id']}>", inline=True)
            if data.get('context'):
                ctx = data['context']
                embed.add_field(name="Context", value=(ctx[:1020] + '...') if len(ctx) > 1024 else ctx, inline=False)
            await message.channel.send(embed=embed)
            return
        # Reverse offender enforcement commands
        if cmd == 'unshadow':
            if not args:
                await message.channel.send("Please specify a user to unshadow. Use .help for commands.")
                return
            # Extract numeric ID from mention or ID string
            import re
            user_id_str = re.sub(r'\D', '', args[0])
            try:
                uid = int(user_id_str)
            except ValueError:
                await message.channel.send("Invalid user ID.")
                return
            if uid in self.shadow_blocked:
                self.shadow_blocked.remove(uid)
                await message.channel.send(f"User <@{uid}> is no longer shadow-blocked.")
            else:
                await message.channel.send(f"User <@{uid}> is not shadow-blocked.")
            return
        if cmd == 'unblock':
            if not args:
                await message.channel.send("Please specify a user to unblock. Use .help for commands.")
                return
            import re
            user_id_str = re.sub(r'\D', '', args[0])
            try:
                uid = int(user_id_str)
            except ValueError:
                await message.channel.send("Invalid user ID.")
                return
            if uid in self.blocked_users:
                self.blocked_users.remove(uid)
                await message.channel.send(f"User <@{uid}> is no longer blocked.")
            else:
                await message.channel.send(f"User <@{uid}> is not blocked.")
            return
        if not args:
            await message.channel.send("Please specify a report ID. Use .help for commands.")
            return
        # Parse report ID
        try:
            report_id = int(args[0])
        except ValueError:
            await message.channel.send("Report ID must be a number.")
            return
        if report_id not in self.report_store:
            await message.channel.send(f"Report ID {report_id} not found.")
            return
        report = self.report_store[report_id]
        guild = self.get_guild(report['guild_id'])
        # Dismiss command
        if cmd == 'dismiss':
            # Mark as dismissed without removing
            self.report_store[report_id]['status'] = 'dismissed'
            await message.channel.send(f"Report {report_id} dismissed (no action). Report retained for further actions.")
            return
        # Remove offending message
        if cmd == 'remove':
            try:
                channel = guild.get_channel(report['channel_id'])
                msg = await channel.fetch_message(report['message_id'])
                await msg.delete()
                await message.channel.send(f"Offending message deleted for report {report_id}.")
            except Exception as e:
                await message.channel.send(f"Error deleting message: {e}")
            self.report_store[report_id]['status'] = 'removed'
            return
        # Warn the user via DM
        if cmd == 'warn':
            reason = ' '.join(args[1:]) if len(args) > 1 else None
            warning_text = f"Your message was flagged as {report['abuse_type']}."
            if reason:
                warning_text += f" Reason: {reason}"
            # Attempt to fetch the user object directly
            offender = None
            try:
                offender = await self.fetch_user(report['offender_id'])
            except Exception:
                pass
            if offender:
                try:
                    await offender.send(warning_text)
                except discord.Forbidden:
                    pass
                await message.channel.send(f"Warning sent to {offender.mention} for report {report_id}.")
            else:
                await message.channel.send("Could not find user to warn.")
            self.report_store[report_id]['status'] = 'warned'
            return
        # Shadow block: hide user's messages
        if cmd == 'shadow_block':
            self.shadow_blocked.add(report['offender_id'])
            self.report_store[report_id]['status'] = 'shadow_blocked'
            await message.channel.send(f"User <@{report['offender_id']}> has been shadow-blocked. Report retained.")
            return
        # Block: add to block-list and notify user
        if cmd == 'block':
            self.blocked_users.add(report['offender_id'])
            # Notify user via DM
            offender = None
            try:
                offender = await self.fetch_user(report['offender_id'])
            except Exception:
                pass
            if offender:
                try:
                    await offender.send("You have been blocked from sending messages here.")
                except discord.Forbidden:
                    pass
            self.report_store[report_id]['status'] = 'blocked'
            await message.channel.send(f"User <@{report['offender_id']}> has been blocked. Report retained.")
            return
        # Escalate to higher admin
        if cmd == 'escalate':
            self.report_store[report_id]['status'] = 'escalated'
            await message.channel.send(f"Report {report_id} escalated to higher admin. Report retained.")
            return
        # Unknown command
        await message.channel.send("Unknown command. Use .help for a list of moderator commands.")

    async def on_reaction_add(self, reaction, user):
        """When a user reacts with 🚩, start DM-based report flow using our new user flow."""
        if user.id == self.user.id:
            return
        channel = reaction.message.channel
        # Only in main group channel
        if channel.name != f'group-{self.group_num}':
            return
        # Use 🚩 emoji as report trigger
        if str(reaction.emoji) != '🚩':
            return

        # Start the reporting flow
        try:
            dm_channel = await user.create_dm()
        except discord.Forbidden:
            return

        # Create a new Report instance with the reported message
        reported_msg = reaction.message
        self.reports[user.id] = Report(self, user, reported_msg)
        # Send initial prompt with numeric options
        await dm_channel.send(
            "Thanks for helping us protect our community!\n"
            "What type of abuse are you reporting?\n"
            "1) Imminent Danger\n"
            "2) Hate Speech\n"
            "3) Explicit Content\n"
            "Reply with the number of your choice."
        )

    async def take_action(self, severity, message):
        """
        Perform automated enforcement actions based on severity level.
        """
        offender_id = message.author.id
        # Mild hate: delete + warning
        if severity == 1:
            await message.delete()
            try:
                await message.author.send(
                    "Your message has been removed for mild hate speech (Severity 1). Please avoid derogatory language."
                )
            except discord.Forbidden:
                pass
            self.violation_history[offender_id] = self.violation_history.get(offender_id, 0) + 1
        # Moderate hate: delete + shadow-block
        elif severity == 2:
            await message.delete()
            self.shadow_blocked.add(offender_id)
            self.violation_history[offender_id] = self.violation_history.get(offender_id, 0) + 1
            mod_channel = self.mod_channels.get(message.guild.id)
            if mod_channel:
                await mod_channel.send(
                    f"Automated detection: User <@{offender_id}> shadow-blocked for 24h (Severity 2: Moderate Hate). Original message: {message.jump_url}"
                )
        # Severe hate: delete + block
        elif severity == 3:
            await message.delete()
            self.blocked_users.add(offender_id)
            self.violation_history[offender_id] = self.violation_history.get(offender_id, 0) + 1
            mod_channel = self.mod_channels.get(message.guild.id)
            if mod_channel:
                await mod_channel.send(
                    f"Automated detection: User <@{offender_id}> blocked (Severity 3: Severe Hate). Original message: {message.jump_url}"
                )
        # Extremist hate: delete + permanent block + escalate
        elif severity == 4:
            await message.delete()
            self.blocked_users.add(offender_id)
            self.violation_history[offender_id] = self.violation_history.get(offender_id, 0) + 1
            mod_channel = self.mod_channels.get(message.guild.id)
            if mod_channel:
                await mod_channel.send(
                    f"Automated detection: User <@{offender_id}> permanently blocked & escalated (Severity 4: Extremist Hate). Original message: {message.jump_url}"
                )

client = ModBot()
client.run(discord_token)