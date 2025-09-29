# Part 2: 基于第一部分的 ER 模型，编写 SQL 语句创建关系模型。（50 分）


```sql
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
```

​1.​主键约束​​：所有表都定义了主键，确保每个记录的唯一性。
- Customer(cid), Company(cid), Individual(cid), Account(aid), Owns(cid, aid), Branch(branch_number), Loan(loan_number), LoanPayment(loan_number, payment_number), Borrows(cid, loan_number).

​2.​外键约束​​：通过FOREIGN KEY实现了引用完整性，并使用ON DELETE CASCADE确保删除操作时相关记录也被删除。
- Company(cid)引用Customer(cid)
- Individual(cid)引用Customer(cid)
- Owns(cid)引用Customer(cid)
- Owns(aid)引用Account(aid)
- Loan(branch_number)引用Branch(branch_number)
- LoanPayment(loan_number)引用Loan(loan_number)
- Borrows(cid)引用Customer(cid)
- Borrows(loan_number)引用Loan(loan_number)

​​3.域约束​​：使用CHECK约束限制了字段值的范围。
- Customer(customer_type)必须为'Company'或'Individual'
- Individual(age)必须非负
- Account(overdraft_limit)必须非负
- Loan(amount)必须非负
- LoanPayment(amount)必须非负

4.​​参与约束​​：使用NOT NULL确保必需字段不为空。
- Customer(cname)和customer_type不为空
- Company(street)和city不为空
- Owns(cid), aid, start_date, pin不为空
- Loan(loan_type), amount, branch_number不为空
- LoanPayment(payment_date), amount不为空
- Borrows(cid), loan_number不为空

​​5.复合键约束​​：Owns, LoanPayment,和Borrows表使用复合主键，确保组合唯一性。
- Owns的主键是(cid, aid)
- LoanPayment的主键是(loan_number, payment_number)
- Borrows的主键是(cid, loan_number)

​​6.子类型处理​​：通过customer_type字段和单独的表（Company和Individual）实现了子类型，并确保每个子类型记录对应一个超类型记录。

## **无法用SQL表示的约束**
尽管SQL语句实现了大多数约束，但以下约束无法完全用SQL语句表示，需要触发器或应用程序逻辑来强制：
- 子类型的互斥约束​​
  - 约束内容​​：每个客户必须是公司或个人之一，但不能同时是两者。即，一个客户不能在Company和Individual表中同时存在记录。

  - 无法表示的原因​​：SQL不支持跨表的互斥约束。虽然customer_type字段通过CHECK约束限制了值，但无法防止同一个cid同时插入到Company和Individual表中。这需要触发器来检查插入或更新操作，确保数据一致性。

- 账户必须被恰好一个客户拥有​​：
  - 约束内容​​：每个账户必须被一个客户拥有（即，每个aid必须在Owns表中出现一次）。
  - 无法表示的原因​​：虽然Owns表中的aid是外键，但SQL无法强制每个aid都必须出现在Owns表中（即，无法强制“每个账户必须被拥有”）。此外，为了确保每个账户只被一个客户拥有，需要在Owns表的aid上添加UNIQUE约束（如SQL语句中的注释建议），但即使如此，也无法强制账户必须被拥有。如果要求账户必须被拥有，需要应用程序逻辑或触发器来确保Account表中的每个aid都在Owns表中。

- 贷款必须被至少一个客户借入​​：
  - 约束内容​​：每笔贷款必须被至少一个客户借入（即，每个loan_number必须在Borrows表中出现至少一次）。
  - 无法表示的原因​​：SQL无法强制外键引用必须存在（即，无法强制Loan表中的每个loan_number都必须在Borrows表中）。这会导致循环引用问题，因为Loan表必须先存在记录才能被Borrows引用，但Borrows表又要求Loan记录存在。因此，需要应用程序逻辑或触发器来确保每笔贷款都被借入。

- 客户类型与子表的一致性​​：
  - 约束内容​​：Customer表中的customer_type必须与子表（Company或Individual）中的记录一致。例如，如果customer_type是'Company'，则该客户必须在Company表中有一条记录，而不在Individual表中。
  - 无法表示的原因​​：SQL无法强制这种跨表的一致性约束。需要触发器在插入或更新时验证customer_type与子表记录匹配。

## 总结
提供的SQL语句成功实现了ER模型中的大多数约束，包括主键、外键、域约束和参与约束。然而，对于子类型的互斥性、强制参与约束（如账户必须被拥有、贷款必须被借入）以及类型一致性，无法仅用SQL语句表示，需要借助触发器或应用程序逻辑来确保完整性和一致性。在实际应用中，建议添加这些触发器或在前端/中间层实现验证逻辑。