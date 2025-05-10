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

        # Forward the message to the mod channel
        mod_channel = self.mod_channels[message.guild.id]
        await mod_channel.send(f'Forwarded message:\n{message.author.name}: "{message.content}"')
        scores = self.eval_text(message.content)
        await mod_channel.send(self.code_format(scores))

    
    def eval_text(self, message):
        ''''
        TODO: Once you know how you want to evaluate messages in your channel, 
        insert your code here! This will primarily be used in Milestone 3. 
        '''
        return message

    
    def code_format(self, text):
        ''''
        TODO: Once you know how you want to show that a message has been 
        evaluated, insert your code here for formatting the string to be 
        shown in the mod channel. 
        '''
        return "Evaluated: '" + text+ "'"

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
                ".dismiss <report_id> - No action\n"
                ".remove <report_id> - Delete the offending message\n"
                ".warn <report_id> [reason] - Send a warning to the user\n"
                ".shadow_block <report_id> - Hide user's messages\n"
                ".block <report_id> - Block user from messaging\n"
                ".escalate <report_id> - Escalate to higher admin\n"
                ".help - Show this help message\n"
            )
            await message.channel.send(usage)
        if cmd == 'help':
            await send_usage()
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
            del self.report_store[report_id]
            await message.channel.send(f"Report {report_id} dismissed (no action).")
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
            del self.report_store[report_id]
            return
        # Warn the user via DM
        if cmd == 'warn':
            reason = ' '.join(args[1:]) if len(args) > 1 else None
            offender = guild.get_member(report['offender_id'])
            warning_text = f"Your message was flagged as {report['abuse_type']}."
            if reason:
                warning_text += f" Reason: {reason}"
            if offender:
                try:
                    await offender.send(warning_text)
                    await message.channel.send(f"Warning sent to {offender.mention} for report {report_id}.")
                except discord.Forbidden:
                    await message.channel.send("Could not DM the user.")
            else:
                await message.channel.send("User not found in guild.")
            del self.report_store[report_id]
            return
        # Shadow block: hide user's messages
        if cmd == 'shadow_block':
            self.shadow_blocked.add(report['offender_id'])
            await message.channel.send(f"User <@{report['offender_id']}> has been shadow-blocked.")
            del self.report_store[report_id]
            return
        # Block: remove and notify user
        if cmd == 'block':
            self.blocked_users.add(report['offender_id'])
            offender = guild.get_member(report['offender_id'])
            if offender:
                try:
                    await offender.send("You have been blocked from sending messages here.")
                except discord.Forbidden:
                    pass
            await message.channel.send(f"User <@{report['offender_id']}> has been blocked.")
            del self.report_store[report_id]
            return
        # Escalate to higher admin
        if cmd == 'escalate':
            await message.channel.send(f"Report {report_id} escalated to higher admin.")
            del self.report_store[report_id]
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
        # Send initial prompt
        await dm_channel.send(
            "Thanks for helping us protect our community! "
            "Please state the kind of danger or abuse you're reporting.\n"
            "Options:\n"
            "• Imminent Danger\n"
            "• Hate Speech\n"
            "• Explicit Content"
        )

client = ModBot()
client.run(discord_token)