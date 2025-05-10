from enum import Enum, auto
import discord
import re

class State(Enum):
    REPORT_START = auto()
    AWAITING_MESSAGE = auto()
    MESSAGE_IDENTIFIED = auto()
    AWAITING_ABUSE_TYPE = auto()
    AWAITING_SUBTYPE = auto()
    AWAITING_CONTEXT = auto()
    AWAITING_CONFIRM = auto()
    REPORT_COMPLETE = auto()

class Report:
    START_KEYWORD = "report"
    CANCEL_KEYWORD = "cancel"
    HELP_KEYWORD = "help"

    def __init__(self, client):
        self.state = State.REPORT_START
        self.client = client
        # Reporter and report data
        self.reporter = None
        self.reported_message = None
        self.report_link = None
        self.abuse_type = None
        self.subtype = None
        self.context = None
        self.message = None
    
    async def handle_message(self, message):
        '''
        This function makes up the meat of the user-side reporting flow. It defines how we transition between states and what 
        prompts to offer at each of those states. You're welcome to change anything you want; this skeleton is just here to
        get you started and give you a model for working with Discord. 
        '''

        if message.content == self.CANCEL_KEYWORD:
            self.state = State.REPORT_COMPLETE
            return ["Report cancelled."]
        
        if self.state == State.REPORT_START:
            # Initialize reporter and prompt for message link
            self.reporter = message.author
            reply = "Thank you for keeping our community safe! Please note that false reporting may lead to disciplinary action.\n\n"
            reply += "Please copy paste the link to the message you want to report.\n"
            reply += "You can obtain this link by right-clicking the message and clicking `Copy Message Link`."
            self.state = State.AWAITING_MESSAGE
            return [reply]
        
        if self.state == State.AWAITING_MESSAGE:
            # Parse out the three ID strings from the message link
            m = re.search('/(\d+)/(\d+)/(\d+)', message.content)
            if not m:
                return ["I'm sorry, I couldn't read that link. Please try again or say `cancel` to cancel."]
            guild = self.client.get_guild(int(m.group(1)))
            if not guild:
                return ["I cannot accept reports of messages from guilds that I'm not in. Please have the guild owner add me to the guild and try again."]
            channel = guild.get_channel(int(m.group(2)))
            if not channel:
                return ["It seems this channel was deleted or never existed. Please try again or say `cancel` to cancel."]
            try:
                reported_msg = await channel.fetch_message(int(m.group(3)))
            except discord.errors.NotFound:
                return ["It seems this message was deleted or never existed. Please try again or say `cancel` to cancel."]

            # Store the reported message and prompt for abuse type
            self.reported_message = reported_msg
            self.report_link = reported_msg.jump_url
            self.state = State.AWAITING_ABUSE_TYPE
            return [
                "I found this message:",
                f"```{reported_msg.author.name}: {reported_msg.content}```",
                "Which abuse type are you reporting? Options: hate speech, harassment, explicit content, misinformation, other."
            ]
        
        # Detailed reporting flow for hate speech
        if self.state == State.AWAITING_ABUSE_TYPE:
            choice = message.content.lower()
            valid = ["hate speech", "harassment", "explicit content", "misinformation", "other"]
            if choice not in valid:
                return ["Please choose one of: hate speech, harassment, explicit content, misinformation, other."]
            self.abuse_type = choice
            if choice != "hate speech":
                await self.forward_to_mod()
                self.state = State.REPORT_COMPLETE
                return [f"Thank you! Your {choice} report has been submitted to our moderators for review."]
            self.state = State.AWAITING_SUBTYPE
            return ["Please specify the subtype of hate speech: racial slurs, religious hate, transphobia, other."]

        if self.state == State.AWAITING_SUBTYPE:
            subtype = message.content.lower()
            valid_subs = ["racial slurs", "religious hate", "transphobia", "other"]
            if subtype not in valid_subs:
                return ["Please choose one of: racial slurs, religious hate, transphobia, other."]
            self.subtype = subtype
            self.state = State.AWAITING_CONTEXT
            return ["Would you like to provide additional context or attachments? Send your message now or type 'skip' to skip."]

        if self.state == State.AWAITING_CONTEXT:
            if message.content.lower() == "skip" and not message.attachments:
                self.context = None
            else:
                self.context = message.content
                if message.attachments:
                    self.context += "\nAttachments:\n" + "\n".join(a.url for a in message.attachments)
            self.state = State.AWAITING_CONFIRM
            summary = f"Here is your report summary:\nMessage: {self.report_link}\nType: {self.abuse_type}"
            if self.subtype:
                summary += f"\nSubtype: {self.subtype}"
            if self.context:
                summary += f"\nContext: {self.context}"
            summary += "\n\nType 'confirm' to submit your report or 'cancel' to cancel."
            return [summary]

        if self.state == State.AWAITING_CONFIRM:
            cmd = message.content.lower()
            if cmd == "confirm":
                await self.forward_to_mod()
                self.state = State.REPORT_COMPLETE
                return ["Thank you! Your report has been submitted. Our moderators will review it shortly."]
            elif cmd == "cancel":
                self.state = State.REPORT_COMPLETE
                return ["Report cancelled."]
            else:
                return ["Please type 'confirm' to submit your report or 'cancel' to cancel."]

    async def forward_to_mod(self):
        """Forward the completed report to the moderator channel as an embed."""
        guild = self.reported_message.guild
        mod_channel = self.client.mod_channels.get(guild.id)
        if mod_channel:
            # Assign unique report ID and store metadata
            report_id = self.client.next_report_id
            self.client.next_report_id += 1
            self.client.report_store[report_id] = {
                'report_id': report_id,
                'reporter_id': self.reporter.id,
                'abuse_type': self.abuse_type,
                'subtype': self.subtype,
                'report_link': self.report_link,
                'offender_id': self.reported_message.author.id,
                'guild_id': guild.id,
                'channel_id': self.reported_message.channel.id,
                'message_id': self.reported_message.id,
                'context': self.context
            }
            embed = discord.Embed(title=f"New Report #{report_id}", color=discord.Color.red())
            embed.set_footer(text=f"Report ID: {report_id}")
            embed.add_field(name="Reporter", value=self.reporter.mention, inline=True)
            embed.add_field(name="Abuse Type", value=self.abuse_type, inline=True)
            if self.subtype:
                embed.add_field(name="Subtype", value=self.subtype, inline=True)
            embed.add_field(name="Message Link", value=self.report_link, inline=False)
            embed.add_field(name="Offending User", value=self.reported_message.author.mention, inline=True)
            if self.context:
                embed.add_field(name="Context", value=(self.context[:1020] + '...') if len(self.context) > 1024 else self.context, inline=False)
            await mod_channel.send(embed=embed)
        else:
            print(f"Mod channel not found for guild {guild.id}")

    def report_complete(self):
        return self.state == State.REPORT_COMPLETE
    


    

