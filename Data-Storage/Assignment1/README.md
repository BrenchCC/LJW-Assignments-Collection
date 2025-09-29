# Assignment 1 - Data Storage
## Part 1: ER Modeling (50 Marks)
- Submit a diagram drawn using a drawing program (any program you prefer is acceptable; you may also use ER drawing tools from http://en.wikipedia.org/wiki/Entity-relationship_model). Hand-drawn diagrams are not allowed.
- If you are uncertain about some of your design choices, you may add explanations.
- Upload your diagram and optional explanations as a single PDF file named `assignment1.pdf`.

- Draw a single ER diagram to represent the specifications listed below.

1.  A banking enterprise needs to store information about **customers** (identified by `cid`, with attribute `cname`) and **accounts** (identified by `aid`, with an `overdraft limit` amount attribute). (8 marks)
2.  A customer is either a **company** (with attributes `street`, `city`) or an **individual** (with attributes `gender`, `age`). (10 marks)
3.  Customers **own** accounts. For each account owned by a customer, we store the `start date` when the account was opened and the `pin number` granting the customer access to the account. A customer can own multiple accounts, but an account can only be owned by one customer. (7 marks)
4.  Add three more entities:
    -   **Loan**: Has the following attributes: `loan number`, `loan type`, `amount`. Each loan has a unique `loan number`.
    -   **Loan Payment**: Has attributes `date`, `amount`, `payment number`. For a given loan, the `payment number` identifies a unique payment, but payments for different loans may share the same number (e.g., Jack Smith's payment #1 is a unique payment of $100, while Jackie Chan's payment #1 might be a different payment of $1000).
    -   **Branch**: Has attributes `branch number`, `city`, `street`. Each branch has a unique `branch number`. (15 marks)
5.  Each loan is **processed at** exactly one branch. Customers **borrow** loans; a customer can have multiple loans, and a given loan may be associated with multiple customers (e.g., a couple might be co-signers for a mortgage loan). (8 marks)
6.  Demonstrate quality (including submission file format and diagram presentation). (2 marks)

Note: If you feel that not all necessary details are fully specified, please use common sense. If in doubt, clearly explain the assumptions you made and how these assumptions affect your model.


## Part 2: Writing SQL Statements to Convert the ER Model from Part 1 into a Relational Model (50 Marks)
1.  **Create the Relational Schema.** Create a table for each relation needed to represent the banking enterprise. You may first convert the ER model into a relational schema and then convert the relational schema into SQL commands. For example, in Chapter 3.1, the Student table is defined as "Students (sid: string, name: string, login: string, age: integer, gpa: real)". You are **not** required to write down the relational schema in this format—you will **not** be graded on the relational schema itself, but only on the SQL statements that create the tables. We merely suggest that defining the relational schema may be helpful for you. (20 marks)
2.  **Implement Integrity Constraints.** Implement the constraints that are implied or explicitly stated, such as domains (e.g., dno is of type integer), key constraints, foreign key constraints, and participation constraints (using NOT NULL). Be sure to incorporate the constraints mentioned in Part 1 as much as possible. If a constraint mentioned in Part 1 cannot be represented by an SQL statement, describe the content of that constraint and explain why it cannot be represented. (25 marks)
3.  **Demonstrate Quality.** (5 marks)