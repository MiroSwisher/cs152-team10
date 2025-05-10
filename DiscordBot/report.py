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

        # Step 1: Category selection
        text = message.content.strip().lower()
        if self.state == State.SELECT_CATEGORY:
            options = {
                "imminent danger": "imminent danger",
                "hate speech": "hate speech",
                "explicit content": "explicit content"
            }
            if text not in options:
                return ["Please choose one of: Imminent Danger, Hate Speech, Explicit Content."]
            self.abuse_type = options[text]
            if self.abuse_type == "imminent danger":
                self.state = State.SELECT_DANGER_TYPE
                return ["Which type of danger?\n• Suicide or Self-Harm\n• Threat of Violence"]
            elif self.abuse_type == "hate speech":
                self.state = State.SELECT_HATE_SUBTYPE
                return ["Which abuse type are you reporting?\n• Racial Slurs\n• Homophobic Language\n• Sexist Language\n• Other hate/harassment"]
            else:
                self.state = State.SELECT_EXPLICIT_SUBTYPE
                return ["Which category best describes the content?\n• Child Sexual Abuse Material\n• Unwanted Sexual Content\n• Sexual Harassment\n• Other explicit content"]

        # Step 2a: Imminent Danger subtype
        if self.state == State.SELECT_DANGER_TYPE:
            valid = ["suicide or self-harm", "threat of violence"]
            if text not in valid:
                return ["Please choose one of: Suicide or Self-Harm, Threat of Violence."]
            self.subtype = text
            if text == "suicide or self-harm":
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

        # Step 2c: Hate Speech subtype
        if self.state == State.SELECT_HATE_SUBTYPE:
            valid = ["racial slurs", "homophobic language", "sexist language", "other hate/harassment"]
            if text not in valid:
                return ["Please choose one of: Racial Slurs, Homophobic Language, Sexist Language, Other hate/harassment."]
            self.subtype = text
            self.state = State.SELECT_HATE_FILTER
            return ["Want to take further steps to prevent seeing this type of content? (Yes/No)"]

        # Step 2d: Hate Speech filter opt-in
        if self.state == State.SELECT_HATE_FILTER:
            if text not in ["yes", "no"]:
                return ["Please answer 'Yes' or 'No'."]
            self.filter_opt_in = (text == "yes")
            if self.filter_opt_in:
                self.state = State.AWAITING_CONTEXT
                return ["Would you like to filter out messages containing this content? (Yes/No)"]
            else:
                self.state = State.AWAITING_CONTEXT
                return ["Thank you! Would you like to add any comments or attach a screenshot? Type 'skip' to skip."]

        # Step 2e: Explicit Content subtype
        if self.state == State.SELECT_EXPLICIT_SUBTYPE:
            valid = ["child sexual abuse material", "unwanted sexual content", "sexual harassment", "other explicit content"]
            if text not in valid:
                return ["Please choose one of: Child Sexual Abuse Material, Unwanted Sexual Content, Sexual Harassment, Other explicit content."]
            self.subtype = text
            self.state = State.SELECT_EXPLICIT_BLOCK
            return ["Would you like to block this account from contacting you? (Yes/No)"]

        # Step 2f: Explicit Content block opt-in
        if self.state == State.SELECT_EXPLICIT_BLOCK:
            if text not in ["yes", "no"]:
                return ["Please answer 'Yes' or 'No'."]
            self.block_opt_in = (text == "yes")
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
        # Store metadata
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
            'status': 'pending'
        }
        # Build embed
        embed = discord.Embed(title=f"New Report #{report_id}", color=discord.Color.red())
        embed.set_footer(text=f"Report ID: {report_id}")
        embed.add_field(name="Reporter", value=self.reporter.mention, inline=True)
        embed.add_field(name="Category", value=self.abuse_type, inline=True)
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
    


    

