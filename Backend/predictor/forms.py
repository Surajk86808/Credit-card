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

    CARD_TYPE_CHOICES = [
        ("credit", "Credit"),
        ("debit", "Debit"),
        ("prepaid", "Prepaid"),
        ("virtual", "Virtual"),
    ]

    TRANSACTION_TYPE_CHOICES = [
        ("online", "Online"),
        ("in_store", "In-Store/POS"),
        ("atm", "ATM"),
        ("recurring", "Recurring"),
    ]

    COUNTRY_CHOICES = [
        ("US", "United States"),
        ("IN", "India"),
        ("GB", "United Kingdom"),
        ("CA", "Canada"),
        ("AU", "Australia"),
        ("DE", "Germany"),
        ("FR", "France"),
        ("SG", "Singapore"),
        ("AE", "United Arab Emirates"),
        ("JP", "Japan"),
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
            "pattern": "\\d{16}",
        }),
    )
    card_network = forms.ChoiceField(
        label="Card Network",
        choices=CARD_NETWORK_CHOICES,
        widget=forms.Select(attrs={"class": "form-control"}),
    )
    card_type = forms.ChoiceField(
        label="Card Type",
        choices=CARD_TYPE_CHOICES,
        widget=forms.Select(attrs={"class": "form-control"}),
    )
    transaction_type = forms.ChoiceField(
        label="Transaction Type",
        choices=TRANSACTION_TYPE_CHOICES,
        widget=forms.Select(attrs={"class": "form-control"}),
    )
    amount = forms.FloatField(
        label="Transaction Amount",
        min_value=0.0,
        widget=forms.NumberInput(attrs={
            "class": "form-control",
            "step": "0.01",
            "placeholder": "e.g., 199.99",
            "inputmode": "decimal",
        }),
    )
    transaction_time = forms.TimeField(
        label="Transaction Time",
        widget=forms.TimeInput(attrs={
            "class": "form-control",
            "type": "time",
        }),
    )
    transaction_location = forms.ChoiceField(
        label="Transaction Location (Country)",
        choices=COUNTRY_CHOICES,
        widget=forms.Select(attrs={"class": "form-control"}),
    )
    is_international = forms.BooleanField(
        label="Is International?",
        required=False,
        widget=forms.CheckboxInput(attrs={"class": "form-check-input"}),
    )
    merchant_category = forms.ChoiceField(
        label="Merchant Category",
        choices=MERCHANT_CATEGORY_CHOICES,
        widget=forms.Select(attrs={"class": "form-control"}),
    )
    credit_limit = forms.FloatField(
        label="Credit Limit",
        min_value=0.0,
        required=False,
        widget=forms.NumberInput(attrs={"class": "form-control", "step": "0.01", "placeholder": "e.g., 5000"}),
    )
    balance = forms.FloatField(
        label="Current Balance",
        min_value=0.0,
        required=False,
        widget=forms.NumberInput(attrs={"class": "form-control", "step": "0.01", "placeholder": "e.g., 1200"}),
    )

    def clean_card_number(self):
        value = self.cleaned_data["card_number"].replace(" ", "").replace("-", "")
        if not re.fullmatch(r"\d{16}", value):
            raise forms.ValidationError("Card number must be exactly 16 digits.")
        return value