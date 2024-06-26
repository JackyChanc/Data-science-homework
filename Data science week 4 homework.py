import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
SALES – Date, Order_id, Item_id, Customer_id, Quantity, Revenue
ITEMS – Item_id, Item_name, price, department
CUSTOMERS- customer_id, first_name,last_name,Address
#1.#Pull total number of orders that were completed on 18th March 2023.
SELECT COUNT(*) AS total_orders
FROM SALES
WHERE Date = '2023-03-18';

#2.#Pull total number of orders that were completed on 18th March 2023 with the first name ‘John’ and last name Doe’.
SELECT COUNT(*) AS total_orders
FROM SALES s
JOIN CUSTOMERS c ON s.Customer_id = c.customer_id
WHERE s.Date = '2023-03-18'
AND c.first_name = 'John'
AND c.last_name = 'Doe';
#3.#Pull total number of customers that purchased in January 2023 and the average amount spend per customer.
SELECT COUNT(DISTINCT Customer_id) AS total_customers,
       AVG(Revenue) AS average_spent_per_customer
FROM SALES
WHERE Date >= '2023-01-01' AND Date <= '2023-01-31';
#4.#Pull the departments that generated less than $600 in 2022.
SELECT department
FROM ITEMS i
JOIN SALES s ON i.Item_id = s.Item_id
WHERE YEAR(Date) = 2022
GROUP BY department
HAVING SUM(Revenue) < 600;
#5.#What is the most and least revenue we have generated by an order.
SELECT MAX(Revenue) AS max_revenue,
       MIN(Revenue) AS min_revenue
FROM SALES;
#6.#What were the orders that were purchased in our most lucrative order.
SELECT *
FROM SALES
WHERE Order_id = (SELECT Order_id
                  FROM SALES
                  GROUP BY Order_id
                  ORDER BY SUM(Revenue) DESC
                  LIMIT 1);
