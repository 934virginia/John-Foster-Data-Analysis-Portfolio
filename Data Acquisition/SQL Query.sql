DROP TABLE services;

CREATE TABLE services(
	customer_id text PRIMARY KEY NOT NULL,
	internet text NOT NULL,
	phone text NOT NULL,
	multiple text NOT NULL,
	online_security text NOT NULL,
	online_backup text NOT NULL,
	device_protection text NOT NULL,
	tech_support text NOT NULL
	);

ADD CONSTRAINT services_fk
	FOREIGN KEY (customer_id)
	REFERENCES customer(customer_id) DEFERRABLE INITIALLY DEFERRED;

COPY services
FROM 'C:\LabFiles\Services.csv'
DELIMITER ','
CSV HEADER;

SELECT
	l.state,
	(cast(COUNT(CASE WHEN internet != 'None' THEN 1 ELSE NULL END) AS float) / cast(COUNT(*) AS float) * 100) AS internet,
	(cast(COUNT(CASE WHEN phone = 'Yes' THEN 1 ELSE NULL END) AS float) / cast(COUNT(*) AS float) * 100) AS phone,
	(cast(COUNT(CASE WHEN online_security = 'Yes' THEN 1 ELSE NULL END) AS float) / cast(COUNT(*) AS float) * 100) AS online_security,
	(cast(COUNT(CASE WHEN online_backup = 'Yes' THEN 1 ELSE NULL END) AS float) / cast(COUNT(*) AS float) * 100) AS online_backup,
	(cast(COUNT(CASE WHEN device_protection = 'Yes' THEN 1 ELSE NULL END) AS float) / cast(COUNT(*) AS float) * 100) AS device_protection,
	(cast(COUNT(CASE WHEN tech_support = 'Yes' THEN 1 ELSE NULL END) AS float) / cast(COUNT(*) AS float) * 100) AS tech_support
FROM services AS s
JOIN customer AS c USING(customer_id)
JOIN location AS l USING(location_id)
GROUP BY l.state;
--
Johnny Foster
He/His
+1 (907) 952 7883
Pacific Standard Time (PST)
Student ID: 011042388
MSDA Program
Program Mentor: Kirk Kelly
