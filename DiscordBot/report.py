from enum import Enum, auto
import discord
import re

class State(Enum):
    SELECT_CATEGORY = auto()
    SELECT_DANGER_TYPE = auto()
    AWAITING_DANGER_ACK = auto()
    SELECT_HATE_SUBTYPE = auto()
    SELECT_HATE_FILTER = auto()
    SELECT_EXPLICIT_SUBTYPE = auto()
    SELECT_EXPLICIT_BLOCK = auto()
    AWAITING_CONTEXT = auto()
    REPORT_COMPLETE = auto()

class Report:
    CANCEL_KEYWORD = "cancel"
    HELP_KEYWORD = "help"

    def __init__(self, client, reporter, reported_message):
        self.client = client
        self.reporter = reporter
        self.reported_message = reported_message
        self.report_link = reported_message.jump_url
        self.abuse_type = None
        self.subtype = None
        self.filter_opt_in = None
        self.block_opt_in = None
        self.context = None
        self.state = State.SELECT_CATEGORY
    
    async def handle_message(self, message):
        """
        Handle each DM message from the user and advance the report flow per state.
        """

        # Allow cancellation
        if message.content.lower() == self.CANCEL_KEYWORD:
            self.state = State.REPORT_COMPLETE
            return ["Report cancelled."]

        # Step 1: Category selection (numeric)
        text = message.content.strip().lower()
        if self.state == State.SELECT_CATEGORY:
            options = {"1": "imminent danger", "2": "hate speech", "3": "explicit content"}
            if text not in options:
                return ["Please choose one of:\n1) Imminent Danger\n2) Hate Speech\n3) Explicit Content"]
            self.abuse_type = options[text]
            if text == "1":
                self.state = State.SELECT_DANGER_TYPE
                return ["Which type of danger?\n1) Suicide or Self-Harm\n2) Threat of Violence"]
            elif text == "2":
                self.state = State.SELECT_HATE_SUBTYPE
                return ["Which abuse type are you reporting?\n1) Racial Slurs\n2) Homophobic Language\n3) Sexist Language\n4) Other hate/harassment"]
            else:
                self.state = State.SELECT_EXPLICIT_SUBTYPE
                return ["Which category best describes the content?\n1) Child Sexual Abuse Material\n2) Unwanted Sexual Content\n3) Sexual Harassment\n4) Other explicit content"]

        # Step 2a: Imminent Danger subtype (numeric)
        if self.state == State.SELECT_DANGER_TYPE:
            options = {"1": "suicide or self-harm", "2": "threat of violence"}
            if text not in options:
                return ["Please choose one of:\n1) Suicide or Self-Harm\n2) Threat of Violence"]
            self.subtype = options[text]
            if text == "1":
                self.state = State.AWAITING_DANGER_ACK
                return [
                    "If you or someone you know is struggling, please reach out:\n"
                    "– U.S. Suicide Prevention Helpline: 988 or 1-800-273-TALK (8255)\n"
                    "– International: https://findahelpline.com\n\n"
                    "Type 'next' to continue."
                ]
            else:
                self.state = State.AWAITING_CONTEXT
                return ["Thank you! Would you like to add any comments or attach a screenshot? Type 'skip' to skip."]

        # Step 2b: Acknowledge helpline
        if self.state == State.AWAITING_DANGER_ACK:
            if text != "next":
                return ["Please type 'next' to continue."]
            self.state = State.AWAITING_CONTEXT
            return ["Thank you! Would you like to add any comments or attach a screenshot? Type 'skip' to skip."]

        # Step 2c: Hate Speech subtype (numeric)
        if self.state == State.SELECT_HATE_SUBTYPE:
            options = {"1": "racial slurs", "2": "homophobic language", "3": "sexist language", "4": "other hate/harassment"}
            if text not in options:
                return ["Please choose one of:\n1) Racial Slurs\n2) Homophobic Language\n3) Sexist Language\n4) Other hate/harassment"]
            self.subtype = options[text]
            self.state = State.SELECT_HATE_FILTER
            return ["Want to take further steps to prevent seeing this type of content?\n1) Yes\n2) No"]

        # Step 2d: Hate Speech filter opt-in (numeric)
        if self.state == State.SELECT_HATE_FILTER:
            options = {"1": True, "2": False}
            if text not in options:
                return ["Please answer:\n1) Yes\n2) No"]
            self.filter_opt_in = options[text]
            self.state = State.AWAITING_CONTEXT
            if self.filter_opt_in:
                return ["Would you like to filter out messages containing this content?\n1) Yes\n2) No"]
            else:
                return ["Thank you! Would you like to add any comments or attach a screenshot? Type 'skip' to skip."]

        # Step 2e: Explicit Content subtype (numeric)
        if self.state == State.SELECT_EXPLICIT_SUBTYPE:
            options = {"1": "child sexual abuse material", "2": "unwanted sexual content", "3": "sexual harassment", "4": "other explicit content"}
            if text not in options:
                return ["Please choose one of:\n1) Child Sexual Abuse Material\n2) Unwanted Sexual Content\n3) Sexual Harassment\n4) Other explicit content"]
            self.subtype = options[text]
            self.state = State.SELECT_EXPLICIT_BLOCK
            return ["Would you like to block this account from contacting you?\n1) Yes\n2) No"]

        # Step 2f: Explicit Content block opt-in (numeric)
        if self.state == State.SELECT_EXPLICIT_BLOCK:
            options = {"1": True, "2": False}
            if text not in options:
                return ["Please answer:\n1) Yes\n2) No"]
            self.block_opt_in = options[text]
            self.state = State.AWAITING_CONTEXT
            return ["Thank you! Would you like to add any comments? Type 'skip' to skip."]

        # Step 3: Context / attachments
        if self.state == State.AWAITING_CONTEXT:
            if text == "skip" and not message.attachments:
                self.context = None
            else:
                self.context = message.content
                if message.attachments:
                    self.context += "\nAttachments:\n" + "\n".join(a.url for a in message.attachments)
            self.state = State.REPORT_COMPLETE
            await self.forward_to_mod()
            return [
                "Got it! We'll send this report to the moderation team for review. "
                "Possible outcomes include: no action, post removal, legal referral, or ban for repeat violations."
            ]
        return []

    async def forward_to_mod(self):
        """Forward the completed report to the moderator channel as an embed."""
        guild = self.reported_message.guild
        mod_channel = self.client.mod_channels.get(guild.id)
        if not mod_channel:
            print(f"Mod channel not found for guild {guild.id}")
            return
        # Assign unique report ID
        report_id = self.client.next_report_id
        self.client.next_report_id += 1
        # Compute automated severity for reported message
        severity = self.client.eval_text(self.reported_message.content)
        # Store metadata including severity
        self.client.report_store[report_id] = {
            'report_id': report_id,
            'reporter_id': self.reporter.id,
            'abuse_type': self.abuse_type,
            'subtype': self.subtype,
            'filter_opt_in': getattr(self, 'filter_opt_in', None),
            'block_opt_in': getattr(self, 'block_opt_in', None),
            'report_link': self.report_link,
            'offender_id': self.reported_message.author.id,
            'guild_id': guild.id,
            'channel_id': self.reported_message.channel.id,
            'message_id': self.reported_message.id,
            'context': self.context,
            'severity': severity,
            'status': 'pending'
        }
        # Build embed
        embed = discord.Embed(title=f"New Report #{report_id}", color=discord.Color.red())
        embed.set_footer(text=f"Report ID: {report_id}")
        embed.add_field(name="Reporter", value=self.reporter.mention, inline=True)
        embed.add_field(name="Category", value=self.abuse_type, inline=True)
        # Include automated severity level
        embed.add_field(name="Severity", value=f"{severity} ({self.client.severity_labels.get(severity)})", inline=True)
        if self.subtype:
            embed.add_field(name="Subtype", value=self.subtype, inline=True)
        if hasattr(self, 'filter_opt_in'):
            embed.add_field(name="Filter Opt-In", value=str(self.filter_opt_in), inline=True)
        if hasattr(self, 'block_opt_in'):
            embed.add_field(name="Block Opt-In", value=str(self.block_opt_in), inline=True)
        # Add violation history for the offending user
        offender_id = self.reported_message.author.id
        history_count = self.client.violation_history.get(offender_id, 0)
        embed.add_field(name="Culprit history of violations", value=str(history_count), inline=False)
        embed.add_field(name="Message Link", value=self.report_link, inline=False)
        embed.add_field(name="Offending User", value=self.reported_message.author.mention, inline=True)
        if self.context:
            embed.add_field(name="Context", value=(self.context[:1020] + '...') if len(self.context) > 1024 else self.context, inline=False)
        await mod_channel.send(embed=embed)

    def report_complete(self):
        return self.state == State.REPORT_COMPLETE
    


    

