import json
import random
from faker import Faker
from datetime import datetime, timedelta

fake = Faker()
Faker.seed(42)
random.seed(42)

NUM_RECORDS = 1000

# Generate vendors
vendors = []
bank_account_ids = []
for i in range(NUM_RECORDS):
    bank_account_id = str(random.randint(1000, 9999))
    vendor = {
        "vendor_id": f"VEND-{1001 + i:04d}",
        "vendor_name": fake.company(),
        "primary_bank_account": bank_account_id
    }
    vendors.append(vendor)
    bank_account_ids.append((bank_account_id, vendor["vendor_id"]))

# Generate invoices
invoices = []
for i in range(NUM_RECORDS):
    vendor = vendors[i % NUM_RECORDS]
    discount = "2% if paid within 10 days" if random.random() < 0.4 else ""
    date = fake.date_between(start_date='-90d', end_date='-10d')
    invoice = {
        "invoice_id": f"INV-{1001 + i:04d}",
        "vendor_id": vendor["vendor_id"],
        "amount": round(random.uniform(500, 15000), 2),
        "date": date.strftime("%Y-%m-%d"),
        "category": "invoice",
        "discount_terms": discount,
    }
    invoices.append(invoice)

# Generate payments
payments = []
for i in range(NUM_RECORDS):
    invoice = invoices[i]
    payment_date = datetime.strptime(invoice["date"], "%Y-%m-%d") + timedelta(days=random.randint(5, 30))
    payment = {
        "payment_id": f"PAY-{2001 + i:04d}",
        "invoice_id": invoice["invoice_id"],
        "paid_date": payment_date.strftime("%Y-%m-%d"),
        "amount": invoice["amount"],
        "method": random.choice(["ACH", "Wire", "Check"])
    }
    payments.append(payment)

# Generate bankaccounts
bankaccounts = []
statuses = ["active", "inactive", "pending"]
used_ids = set()
for i in range(NUM_RECORDS):
    bank_account_id, vendor_id = bank_account_ids[i]
    while bank_account_id in used_ids:
        bank_account_id = str(random.randint(1000, 9999))
    used_ids.add(bank_account_id)
    change_date = fake.date_between(start_date='-120d', end_date='today')
    bankacc = {
        "bank_account_id": bank_account_id,
        "vendor_id": vendor_id,
        "last_changed": change_date.strftime("%Y-%m-%d"),
        "status": random.choices(statuses, [0.7, 0.2, 0.1])[0]
    }
    bankaccounts.append(bankacc)

# Save to json files
with open('vendors.json', 'w') as f:
    json.dump(vendors, f, indent=2)
with open('invoices.json', 'w') as f:
    json.dump(invoices, f, indent=2)
with open('payments.json', 'w') as f:
    json.dump(payments, f, indent=2)
with open('bankaccounts.json', 'w') as f:
    json.dump(bankaccounts, f, indent=2)

import pandas as pd

# Load the four just-created JSON files back in as DataFrames
vendors_df = pd.read_json('vendors.json')
invoices_df = pd.read_json('invoices.json')
payments_df = pd.read_json('payments.json')
bankaccounts_df = pd.read_json('bankaccounts.json')

# Flat merge (like a DynamoDB denormalized record)
flat = invoices_df.merge(vendors_df, on='vendor_id', how='left')
flat = flat.merge(payments_df, on='invoice_id', how='left', suffixes=('_invoice', '_payment'))

# Merge bank account details for each vendor (primary only)
bank_map = bankaccounts_df[['bank_account_id', 'status', 'last_changed']]
flat = flat.merge(
    bank_map, left_on='primary_bank_account', right_on='bank_account_id', how='left'
)

# Rename and keep only analytics-relevant columns
flat = flat.rename(columns={
    'vendor_name': 'vendor',
    'amount_invoice': 'invoice_amount',
    'amount_payment': 'paid_amount'
})

flat = flat[[
    'invoice_id',
    'vendor_id',
    'vendor',
    'primary_bank_account',
    'status',
    'last_changed',
    'invoice_amount',
    'paid_amount',
    'date',
    'paid_date',
    'discount_terms',
    'category',
    'method'
]]

flat.to_json('all_data_flat.json', orient='records', indent=2)
