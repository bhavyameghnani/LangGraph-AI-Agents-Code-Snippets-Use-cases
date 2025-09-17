import pandas as pd

# Create dummy sensitive data for wealth management (5 rows each)
clients_data = {
    "ClientID": ["C1001", "C1002", "C1003", "C1004", "C1005"],
    "FullName": ["Rahul Sharma", "Ananya Mehta", "Vikram Kapoor", "Sneha Iyer", "Amit Desai"],
    "PAN": ["ABCDE1234F", "FGHIJ5678K", "LMNOP9012Q", "RSTUV3456W", "XYZAB7890C"],
    "Passport": ["P1234567", "P7654321", "P1122334", "P9988776", "P5566778"],
    "Phone": ["9876543210", "9123456780", "9988776655", "9765432109", "9321456789"]
}

accounts_data = {
    "AccountID": ["A5001", "A5002", "A5003", "A5004", "A5005"],
    "ClientID": ["C1001", "C1002", "C1003", "C1004", "C1005"],
    "FullName": ["Rahul Sharma", "Ananya Mehta", "Vikram Kapoor", "Sneha Iyer", "Amit Desai"],
    "PAN": ["ABCDE1234F", "FGHIJ5678K", "LMNOP9012Q", "RSTUV3456W", "XYZAB7890C"],
    "AccountType": ["Savings", "Demat", "Mutual Fund", "Portfolio", "Savings"],
    "Balance": [1250000.75, 845000.50, 1575000.00, 2200000.25, 965000.90],
    "BranchCode": ["B001", "B002", "B001", "B003", "B002"]
}

transactions_data = {
    "TxnID": ["T9001", "T9002", "T9003", "T9004", "T9005"],
    "AccountID": ["A5001", "A5002", "A5003", "A5004", "A5005"],
    "TxnDate": ["2025-09-01", "2025-09-02", "2025-09-03", "2025-09-04", "2025-09-05"],
    "Amount": [25000.00, 50000.75, 75000.50, 125000.00, 100000.25],
    "TxnType": ["Credit", "Debit", "Credit", "Debit", "Credit"],
     "ClientID": ["C1001", "C1002", "C1003", "C1004", "C1005"],
     "Phone": ["9876543210", "9123456780", "9988776655", "9765432109", "9321456789"]
}

# Convert to DataFrames
df_clients = pd.DataFrame(clients_data)
df_accounts = pd.DataFrame(accounts_data)
df_transactions = pd.DataFrame(transactions_data)

# Save as CSV files
clients_file = "input/clients.csv"
accounts_file = "input/accounts.csv"
transactions_file = "input/transactions.csv"

df_clients.to_csv(clients_file, index=False)
df_accounts.to_csv(accounts_file, index=False)
df_transactions.to_csv(transactions_file, index=False)

(clients_file, accounts_file, transactions_file)
