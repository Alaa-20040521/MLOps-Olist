# MLOps Olist — Task 1

## Project Overview

This project uses the Brazilian E-Commerce Public Dataset by Olist to practice the complete MLOps workflow, starting from raw relational data and progressing toward data analysis and machine learning.

Task 1 focuses on loading the Olist dataset into a local PostgreSQL database and verifying the database structure and relationships.

## Dataset

Dataset: Brazilian E-Commerce Public Dataset by Olist

Source: Kaggle

The dataset contains multiple relational CSV files representing:

* Orders
* Customers
* Order Items
* Order Payments
* Order Reviews
* Products
* Sellers
* Geolocation
* Product Category Translation

## Technologies

* PostgreSQL 16
* Docker
* Docker Compose
* SQL
* CSV

## Database Architecture

The main relationships are:

```text
customers
    |
    | customer_id
    v
orders
    |
    | order_id
    v
order_items
    |              |
    | product_id   | seller_id
    v              v
products        sellers

orders
    |
    +---- order_payments
    |
    +---- order_reviews

geolocation
    |
    +---- ZIP-code based location information
```

## Docker Setup

PostgreSQL runs locally inside a Docker container.

Container:

```text
olist-postgres
```

Database:

```text
olist
```

User:

```text
olist_user
```

PostgreSQL version:

```text
16
```

Port:

```text
5432
```

## Data Loading

The CSV files were loaded into PostgreSQL using the PostgreSQL `COPY` command.

The dataset files were mounted into the PostgreSQL container through Docker Compose.

## Tables

The database contains the following tables:

| Table                        |      Rows |
| ---------------------------- | --------: |
| customers                    |    99,441 |
| orders                       |    99,441 |
| order_items                  |   112,650 |
| order_payments               |   103,886 |
| order_reviews                |    99,224 |
| products                     |    32,951 |
| sellers                      |     3,095 |
| geolocation                  | 1,000,163 |
| product_category_translation |        71 |

## SQL Validation

Several SQL queries were executed to verify that the database was working correctly.

Examples include:

* Selecting records from tables
* Counting rows
* Grouping customers by state
* Joining orders with customers
* Joining orders with order items
* Joining order items with products
* Joining order items with sellers
* Aggregating order items at the order level

## Important Database Relationship

An important relationship discovered during the task is:

```text
One Order → Many Order Items
```

Therefore, one order may appear in multiple rows in `order_items`.

This is important for future feature engineering because the final machine learning dataset will generally require one row per order.

## Machine Learning Problem

The recommended supervised learning problem is late-delivery classification.

The objective is to predict whether an order will be delivered:

```text
Late
```

or

```text
On Time
```

The target can be derived by comparing the actual delivery date with the estimated delivery date.

## Data Leakage Consideration

Actual delivery information occurs after the prediction point.

Therefore, actual delivery dates and other future information should not be used as model features when predicting delivery lateness.

They may instead be used to construct the target variable.

## Task 1 Status

Task 1 completed.

Completed:

* Downloaded and inspected the dataset
* Set up PostgreSQL with Docker
* Created relational database tables
* Loaded CSV data into PostgreSQL
* Verified row counts
* Executed SQL queries
* Tested table relationships using JOINs
* Understood one-to-many relationships
* Identified the late-delivery classification problem
* Identified the risk of data leakage