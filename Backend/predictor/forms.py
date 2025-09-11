from django import forms
import re

# Placeholder for dynamic fallback when features are unavailable
class FallbackPredictForm(forms.Form):
    CARD_NETWORK_CHOICES = [
        ("visa", "Visa"),
        ("mastercard", "Mastercard"),
        ("amex", "American Express"),
        ("discover", "Discover"),
        ("rupay", "RuPay"),
    ]

    TRANSACTION_TYPE_CHOICES = [
        ("online", "Online"),
        ("in_store", "In-Store/POS"),
        ("atm", "ATM"),
        ("recurring", "Recurring"),
    ]

    LOCATION_TYPE_CHOICES = [
        ("domestic", "Domestic"),
        ("international", "International"),
    ]

    DEVICE_TYPE_CHOICES = [
        ("mobile", "Mobile"),
        ("desktop", "Desktop"),
        ("card_reader", "Card Reader/Terminal"),
    ]

    MERCHANT_CATEGORY_CHOICES = [
        ("grocery", "Grocery"),
        ("fuel", "Fuel"),
        ("electronics", "Electronics"),
        ("travel", "Travel"),
        ("dining", "Dining"),
        ("other", "Other"),
    ]

    card_number = forms.CharField(
        label="Card Number",
        min_length=16,
        max_length=19,
        help_text="Enter 16-digit card number (digits only)",
        widget=forms.TextInput(attrs={
            "class": "form-control",
            "placeholder": "1234 5678 9012 3456",
            "inputmode": "numeric",
            "pattern": "\\d{16,19}",
        }),
    )
    card_network = forms.ChoiceField(
        label="Card Network",
        choices=CARD_NETWORK_CHOICES,
        widget=forms.Select(attrs={"class": "form-control"}),
    )
    transaction_type = forms.ChoiceField(
        label="Transaction Type",
        choices=TRANSACTION_TYPE_CHOICES,
        widget=forms.Select(attrs={"class": "form-control"}),
    )
    amount = forms.FloatField(
        label="Amount",
        min_value=0.0,
        widget=forms.NumberInput(attrs={"class": "form-control", "step": "0.01", "placeholder": "e.g., 199.99"}),
    )
    time_of_day = forms.IntegerField(
        label="Hour of Day (0-23)",
        min_value=0,
        max_value=23,
        widget=forms.NumberInput(attrs={"class": "form-control", "placeholder": "e.g., 14"}),
    )
    merchant_category = forms.ChoiceField(
        label="Merchant Category",
        choices=MERCHANT_CATEGORY_CHOICES,
        widget=forms.Select(attrs={"class": "form-control"}),
    )
    location_type = forms.ChoiceField(
        label="Location Type",
        choices=LOCATION_TYPE_CHOICES,
        widget=forms.Select(attrs={"class": "form-control"}),
    )
    device_type = forms.ChoiceField(
        label="Device Type",
        choices=DEVICE_TYPE_CHOICES,
        widget=forms.Select(attrs={"class": "form-control"}),
    )
    customer_age = forms.IntegerField(
        label="Customer Age",
        min_value=16,
        max_value=100,
        widget=forms.NumberInput(attrs={"class": "form-control", "placeholder": "e.g., 32"}),
    )
    credit_limit = forms.FloatField(
        label="Credit Limit",
        min_value=0.0,
        widget=forms.NumberInput(attrs={"class": "form-control", "step": "0.01", "placeholder": "e.g., 5000"}),
    )
    balance = forms.FloatField(
        label="Current Balance",
        min_value=0.0,
        widget=forms.NumberInput(attrs={"class": "form-control", "step": "0.01", "placeholder": "e.g., 1200"}),
    )

    def clean_card_number(self):
        value = self.cleaned_data["card_number"].replace(" ", "").replace("-", "")
        if not re.fullmatch(r"\d{16}", value):
            raise forms.ValidationError("Card number must be exactly 16 digits.")
        return value