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
from classifier import load_model, predict_severity, tuned_llm_classification
from vertexai import init
from vertexai.generative_models import GenerativeModel
from hate_speech_classifier import HateSpeechClassifier

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
    project_id = tokens['PROJECT']
    region = tokens['REGION']
    endpoint = tokens['ENDPOINT']

# Initialize Vertex AI
init(project=project_id, location=region)
tuned_model = GenerativeModel(model_name=endpoint)

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
        # Load traditional classifier
        self.vectorizer, self.traditional_classifier = load_model()
        self.llm_classifier = HateSpeechClassifier(project_id=os.getenv('GOOGLE_CLOUD_PROJECT'), location='us-central1')
        # Define severity labels
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
        is_hate, confidence, severity = self.eval_text(message.content)
        await self.take_action(is_hate, confidence, message)

        # Forward the message to the mod channel with severity info
        mod_channel = self.mod_channels[message.guild.id]
        await mod_channel.send(f'Forwarded message:\n{message.author.name}: "{message.content}"')
        await mod_channel.send(self.code_format(is_hate, confidence, severity))

    def eval_text(self, message):
        """
        Run the combined hate speech classification system.
        Returns severity level and confidence.
        """
        try:
            # Get severity from traditional classifier
            vectorizer, clf = self.vectorizer, self.traditional_classifier
            severity = predict_severity(message, vectorizer, clf)
            
            # Get tuned LLM prediction
            prompt = f"""Analyze this text for hate speech: {message}

Please classify the severity of hate speech in this text on a scale from 0-4:
0: Non-Hateful
1: Mild Hate (animosity)
2: Moderate Hate (derogation, dehumanization)
3: Severe Hate (threatening)
4: Extremist Hate (support for hate)

Return ONLY a single number (0-4) representing the severity level, with no additional text or explanation."""

            # Get response from tuned model
            response = tuned_model.generate_content(
                prompt,
                generation_config={"temperature": 0.2, "max_output_tokens": 128}
            )
            
            # Parse response - expect just a number
            try:
                # Clean and validate response
                response_text = response.text.strip()
                if not response_text:
                    logger.warning(f"Empty response from tuned LLM for text: {message[:100]}...")
                    llm_severity = 0
                else:
                    # Try to extract a number from the response
                    numbers = re.findall(r'\d+', response_text)
                    if numbers:
                        llm_severity = int(numbers[0])
                        if not 0 <= llm_severity <= 4:
                            logger.warning(f"Invalid severity {llm_severity} from tuned LLM for text: {message[:100]}...")
                            llm_severity = 0
                    else:
                        logger.warning(f"No number found in response: {response_text}\nText: {message[:100]}...")
                        llm_severity = 0
            except Exception as e:
                logger.error(f"Failed to parse tuned LLM response: {response_text}\nError: {str(e)}")
                llm_severity = 0
            
            # Take max severity between both classifiers
            final_severity = max(severity, llm_severity)
            
            # Determine if it's hate speech based on severity
            is_hate = final_severity > 0
            confidence = 'high' if severity == llm_severity else 'medium'
            
            return is_hate, confidence, final_severity
            
        except Exception as e:
            logger.error(f"Error in eval_text: {str(e)}")
            return False, 'low', 0

    def code_format(self, is_hate, confidence, severity):
        """
        Format the classifier output for the mod channel.
        """
        label = self.severity_labels.get(severity, "Unknown")
        return f"Automated detection: {'Hate Speech' if is_hate else 'Non-Hate'} (Severity: {severity} - {label}, Confidence: {confidence})"

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
                await message.channel.send(f"User <@{uid}> was not shadow-blocked.")
            return
        if cmd == 'unblock':
            if not args:
                await message.channel.send("Please specify a user to unblock. Use .help for commands.")
                return
            # Extract numeric ID from mention or ID string
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
                await message.channel.send(f"User <@{uid}> was not blocked.")
            return
        # Report action commands
        if cmd in ['dismiss', 'remove', 'warn', 'shadow_block', 'block', 'escalate']:
            if not args:
                await message.channel.send(f"Please specify a report ID for {cmd}. Use .help for commands.")
                return
            try:
                rid = int(args[0])
            except ValueError:
                await message.channel.send("Report ID must be an integer.")
                return
            if rid not in self.report_store:
                await message.channel.send(f"Report ID {rid} not found.")
                return
            data = self.report_store[rid]
            if data.get('status') != 'pending':
                await message.channel.send(f"Report #{rid} has already been resolved.")
                return
            # Get the message link and offender ID
            msg_link = data['report_link']
            offender_id = data['offender_id']
            # Take the requested action
            if cmd == 'dismiss':
                data['status'] = 'dismissed'
                await message.channel.send(f"Report #{rid} dismissed.")
            elif cmd == 'remove':
                # Extract message ID from the link
                msg_id = int(msg_link.split('/')[-1])
                try:
                    msg = await message.channel.fetch_message(msg_id)
                    await msg.delete()
                    data['status'] = 'removed'
                    await message.channel.send(f"Message removed. Report #{rid} marked as resolved.")
                except discord.NotFound:
                    await message.channel.send("Message not found. It may have been deleted already.")
                except discord.Forbidden:
                    await message.channel.send("I don't have permission to delete that message.")
            elif cmd == 'warn':
                reason = ' '.join(args[1:]) if len(args) > 1 else "No reason provided"
                try:
                    offender = await self.fetch_user(offender_id)
                    await offender.send(
                        f"You have been warned for violating our community guidelines.\n"
                        f"Reason: {reason}\n"
                        f"Please review our rules and avoid further violations."
                    )
                    data['status'] = 'warned'
                    await message.channel.send(f"Warning sent to <@{offender_id}>. Report #{rid} marked as resolved.")
                except discord.Forbidden:
                    await message.channel.send("Could not send warning DM to the user.")
            elif cmd == 'shadow_block':
                self.shadow_blocked.add(offender_id)
                data['status'] = 'shadow_blocked'
                await message.channel.send(f"User <@{offender_id}> is now shadow-blocked. Report #{rid} marked as resolved.")
            elif cmd == 'block':
                self.blocked_users.add(offender_id)
                data['status'] = 'blocked'
                await message.channel.send(f"User <@{offender_id}> is now blocked. Report #{rid} marked as resolved.")
            elif cmd == 'escalate':
                data['status'] = 'escalated'
                await message.channel.send(f"Report #{rid} has been escalated to higher administration.")
            return
        # Unknown command
        await message.channel.send("Unknown command. Use .help to see available commands.")

    async def on_reaction_add(self, reaction, user):
        """When a user reacts with 🚩, start DM-based report flow."""
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

    async def take_action(self, is_hate, confidence, message):
        """Take appropriate action based on hate speech detection."""
        offender_id = message.author.id
        
        # Log automated flag as a report entry if hate speech detected
        if is_hate:
            report_id = self.next_report_id
            self.next_report_id += 1
            
            # Store automated report metadata
            self.report_store[report_id] = {
                'report_id': report_id,
                'reporter_id': self.user.id,
                'abuse_type': 'hate speech',
                'confidence': confidence,
                'report_link': message.jump_url,
                'offender_id': offender_id,
                'guild_id': message.guild.id,
                'channel_id': message.channel.id,
                'message_id': message.id,
                'status': 'automated'
            }
            
            # Delete the message
            await message.delete()
            
            # Get mod channel
            mod_channel = self.mod_channels.get(message.guild.id)
            
            # Update violation history
            if offender_id not in self.violation_history:
                self.violation_history[offender_id] = 0
            self.violation_history[offender_id] += 1
            
            # Take action based on confidence and violation history
            if confidence == 'high' or self.violation_history[offender_id] > 2:
                # Block user for high confidence or repeat violations
                self.blocked_users.add(offender_id)
                if mod_channel:
                    await mod_channel.send(
                        f"🚫 Automated detection: User <@{offender_id}> blocked for hate speech "
                        f"(Confidence: {confidence}, Violations: {self.violation_history[offender_id]}). "
                        f"Original message: {message.jump_url}"
                    )
                try:
                    await message.author.send("You have been blocked from sending messages here due to hate speech.")
                except discord.Forbidden:
                    pass
            else:
                # Shadow block for lower confidence
                self.shadow_blocked.add(offender_id)
                if mod_channel:
                    await mod_channel.send(
                        f"⚠️ Automated detection: User <@{offender_id}> shadow-blocked for 24h "
                        f"(Confidence: {confidence}). Original message: {message.jump_url}"
                    )
                try:
                    await message.author.send(
                        "Your message has been removed for hate speech. "
                        "Please avoid harmful language. Further violations may result in being blocked."
                    )
                except discord.Forbidden:
                    pass

# Run the bot
client = ModBot()
client.run(discord_token)