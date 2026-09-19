# Olist MLOps - Task 1 SQL Queries

## 1. Check Customers

```sql
SELECT *
FROM customers
LIMIT 10;
```

## 2. Count Customers

```sql
SELECT COUNT(*) AS total_customers
FROM customers;
```

## 3. Customers by State

```sql
SELECT
    customer_state,
    COUNT(*) AS customer_count
FROM customers
GROUP BY customer_state
ORDER BY customer_count DESC;
```

## 4. Customers by City

```sql
SELECT
    customer_city,
    customer_state,
    COUNT(*) AS customer_count
FROM customers
GROUP BY customer_city, customer_state
ORDER BY customer_count DESC
LIMIT 10;
```

## 5. First JOIN - Orders + Customers

```sql
SELECT
    o.order_id,
    o.customer_id,
    c.customer_city,
    c.customer_state
FROM orders o
JOIN customers c
    ON o.customer_id = c.customer_id
LIMIT 20;
```

## 6. Verify Orders-Customers JOIN

```sql
SELECT COUNT(*)
FROM orders o
JOIN customers c
    ON o.customer_id = c.customer_id;
```

Expected result:

99441

## 7. Second JOIN - Orders + Order Items

```sql
SELECT
    o.order_id,
    oi.product_id,
    oi.seller_id,
    oi.price,
    oi.freight_value
FROM orders o
JOIN order_items oi
    ON o.order_id = oi.order_id
LIMIT 20;
```

## 8. Number of Items per Order

```sql
SELECT
    o.order_id,
    COUNT(*) AS item_count
FROM orders o
JOIN order_items oi
    ON o.order_id = oi.order_id
GROUP BY o.order_id
ORDER BY item_count DESC
LIMIT 20;
```

## 9. Items + Products

```sql
SELECT
    oi.order_id,
    oi.product_id,
    p.product_category_name
FROM order_items oi
JOIN products p
    ON oi.product_id = p.product_id
LIMIT 10;
```

## 10. Items + Sellers

```sql
SELECT
    oi.order_id,
    oi.seller_id,
    s.seller_city,
    s.seller_state
FROM order_items oi
JOIN sellers s
    ON oi.seller_id = s.seller_id
LIMIT 10;
```

## 11. Order-Level Aggregation

```sql
SELECT
    order_id,
    COUNT(*) AS number_of_items,
    SUM(price) AS total_product_value,
    SUM(freight_value) AS total_freight
FROM order_items
GROUP BY order_id;
```

## 12. Final Row Count Verification

```sql
SELECT 'customers' AS table_name, COUNT(*) AS rows FROM customers
UNION ALL
SELECT 'orders', COUNT(*) FROM orders
UNION ALL
SELECT 'order_items', COUNT(*) FROM order_items
UNION ALL
SELECT 'order_payments', COUNT(*) FROM order_payments
UNION ALL
SELECT 'order_reviews', COUNT(*) FROM order_reviews
UNION ALL
SELECT 'products', COUNT(*) FROM products
UNION ALL
SELECT 'sellers', COUNT(*) FROM sellers
UNION ALL
SELECT 'geolocation', COUNT(*) FROM geolocation
UNION ALL
SELECT 'product_category_translation', COUNT(*) FROM product_category_translation;
```

Expected row counts:

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
