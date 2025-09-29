-- Part 2: SQL Implementation for Banking ER Model
-- This script creates all tables and constraints for the banking database
-- Note: SQLite does not enforce foreign keys by default, so we enable it with PRAGMA

PRAGMA foreign_keys = ON;

-- 1. Customer table (supertype)
CREATE TABLE Customer (
    cid INTEGER PRIMARY KEY,
    cname TEXT NOT NULL,
    customer_type TEXT NOT NULL CHECK (customer_type IN ('Company', 'Individual'))
);

-- 2. Company subtype table
CREATE TABLE Company (
    cid INTEGER PRIMARY KEY,
    street TEXT NOT NULL,
    city TEXT NOT NULL,
    FOREIGN KEY (cid) REFERENCES Customer(cid) ON DELETE CASCADE
);

-- 3. Individual subtype table
CREATE TABLE Individual (
    cid INTEGER PRIMARY KEY,
    gender TEXT,
    age INTEGER CHECK (age >= 0),
    FOREIGN KEY (cid) REFERENCES Customer(cid) ON DELETE CASCADE
);

-- 4. Account table
CREATE TABLE Account (
    aid INTEGER PRIMARY KEY,
    overdraft_limit REAL CHECK (overdraft_limit >= 0)
);

-- 5. Owns relationship table (with attributes)
CREATE TABLE Owns (
    cid INTEGER NOT NULL,
    aid INTEGER NOT NULL,
    start_date DATE NOT NULL,
    pin TEXT NOT NULL,
    PRIMARY KEY (cid, aid),
    FOREIGN KEY (cid) REFERENCES Customer(cid) ON DELETE CASCADE,
    FOREIGN KEY (aid) REFERENCES Account(aid) ON DELETE CASCADE
);

-- 6. Branch table
CREATE TABLE Branch (
    branch_number INTEGER PRIMARY KEY,
    city TEXT NOT NULL,
    street TEXT NOT NULL
);

-- 7. Loan table
CREATE TABLE Loan (
    loan_number INTEGER PRIMARY KEY,
    loan_type TEXT NOT NULL,
    amount REAL CHECK (amount >= 0),
    branch_number INTEGER NOT NULL,
    FOREIGN KEY (branch_number) REFERENCES Branch(branch_number)
);

-- 8. LoanPayment table
CREATE TABLE LoanPayment (
    loan_number INTEGER NOT NULL,
    payment_number INTEGER NOT NULL,
    payment_date DATE NOT NULL,
    amount REAL CHECK (amount >= 0),
    PRIMARY KEY (loan_number, payment_number),
    FOREIGN KEY (loan_number) REFERENCES Loan(loan_number) ON DELETE CASCADE
);

-- 9. Borrows relationship table (many-to-many between Customer and Loan)
CREATE TABLE Borrows (
    cid INTEGER NOT NULL,
    loan_number INTEGER NOT NULL,
    PRIMARY KEY (cid, loan_number),
    FOREIGN KEY (cid) REFERENCES Customer(cid) ON DELETE CASCADE,
    FOREIGN KEY (loan_number) REFERENCES Loan(loan_number) ON DELETE CASCADE
);

-- Insert sample data for testing
-- Customers
INSERT INTO Customer (cid, cname, customer_type) VALUES 
(1, 'ABC Corporation', 'Company'),
(2, 'John Smith', 'Individual'),
(3, 'Jane Doe', 'Individual'),
(4, 'XYZ Enterprises', 'Company');

-- Company customers
INSERT INTO Company (cid, street, city) VALUES 
(1, '123 Business Ave', 'New York'),
(4, '456 Commerce St', 'Chicago');

-- Individual customers
INSERT INTO Individual (cid, gender, age) VALUES 
(2, 'Male', 35),
(3, 'Female', 28);

-- Accounts
INSERT INTO Account (aid, overdraft_limit) VALUES 
(101, 1000.00),
(102, 500.00),
(103, 2000.00),
(104, 0.00);

-- Ownership relationships
INSERT INTO Owns (cid, aid, start_date, pin) VALUES 
(1, 101, '2025-01-15', '1234'),
(2, 102, '2025-02-20', '5678'),
(3, 103, '2025-03-10', '9012'),
(1, 104, '2025-04-05', '3456');

-- Branches
INSERT INTO Branch (branch_number, city, street) VALUES 
(1, 'New York', '100 Main St'),
(2, 'Chicago', '200 Oak Ave'),
(3, 'Los Angeles', '300 Sunset Blvd');

-- Loans
INSERT INTO Loan (loan_number, loan_type, amount, branch_number) VALUES 
(1001, 'Mortgage', 250000.00, 1),
(1002, 'Auto', 35000.00, 2),
(1003, 'Personal', 10000.00, 1),
(1004, 'Business', 500000.00, 2);

-- Loan payments
INSERT INTO LoanPayment (loan_number, payment_number, payment_date, amount) VALUES 
(1001, 1, '2025-05-01', 1500.00),
(1001, 2, '2025-06-01', 1500.00),
(1002, 1, '2025-05-15', 500.00),
(1003, 1, '2025-05-20', 300.00);

-- Borrowing relationships
INSERT INTO Borrows (cid, loan_number) VALUES 
(1, 1004),
(2, 1002),
(2, 1003),
(3, 1001),
(3, 1003);

-- Test queries to verify data integrity
-- 1. Check customer data
SELECT * FROM Customer;

-- 2. Check subtype data
SELECT c.cid, c.cname, c.customer_type, co.street, co.city, i.gender, i.age
FROM Customer c
LEFT JOIN Company co ON c.cid = co.cid
LEFT JOIN Individual i ON c.cid = i.cid;

-- 3. Check account ownership
SELECT o.cid, c.cname, o.aid, a.overdraft_limit, o.start_date, o.pin
FROM Owns o
JOIN Customer c ON o.cid = c.cid
JOIN Account a ON o.aid = a.aid;

-- 4. Check loans and their branches
SELECT l.loan_number, l.loan_type, l.amount, b.branch_number, b.city, b.street
FROM Loan l
JOIN Branch b ON l.branch_number = b.branch_number;

-- 5. Check loan payments
SELECT lp.loan_number, l.loan_type, lp.payment_number, lp.payment_date, lp.amount
FROM LoanPayment lp
JOIN Loan l ON lp.loan_number = l.loan_number;

-- 6. Check borrowing relationships
SELECT br.cid, c.cname, br.loan_number, l.loan_type, l.amount
FROM Borrows br
JOIN Customer c ON br.cid = c.cid
JOIN Loan l ON br.loan_number = l.loan_number;

