
-- Database: NewDB

-- DROP DATABASE IF EXISTS "NewDB";
/*
CREATE DATABASE "NewDB"
    WITH
    OWNER = postgres
    ENCODING = 'UTF8'
    LC_COLLATE = 'English_United States.1252'
    LC_CTYPE = 'English_United States.1252'
    TABLESPACE = pg_default
    CONNECTION LIMIT = -1
    IS_TEMPLATE = False;
	*/

/*
--DROP TABLE IF EXISTS CensusData;
	-- Step 2: Import the Census data into the table
CREATE TABLE CensusData (
    -- Define columns similar to the structure of the CSV file
    -- Replace column_names with actual column names from the CSV
    column1 VARCHAR(255),
    column2 VARCHAR(255),
	column3 VARCHAR(255),
	column4 VARCHAR(255),
	column5 VARCHAR(255)
    -- Add more columns as needed
);
*/

-- \copy CensusData(column1, column2, column3, column4, column5) from 'C:\Users\johnf\OneDrive\Desktop\school\sql\ACSST1Y2021.S2801-2023-05-14T010706.csv' DELIMITER ',' CSV HEADER;

-- Step 3: Calculate the tablet ownership percentage
SELECT 
    CAST(REPLACE(REPLACE(REPLACE("column4", ' ', ''), '%', ''), '\n', '') AS FLOAT) AS tablet_percentage
INTO tablet_percentage_table
FROM CensusData
WHERE "column1" = 'Â Â Â Â Â Â Â Â Â Â Â Â Tablet or other portable wireless computer'
LIMIT 1;

-- Step 4: Calculate the counts of surveyed households by income brackets
SELECT 
    CAST(REPLACE("column2", ',', '') AS INT) AS income_count
INTO income_count_table
FROM CensusData
WHERE "column1" IN ('Â Â Â Â Â Â Â Â Less than $20,000:', 'Â Â Â Â Â Â Â Â $20,000 to $74,999:', 'Â Â Â Â Â Â Â Â $75,000 or more:')
ORDER BY CASE 
    WHEN "column1" = 'Â Â Â Â Â Â Â Â Less than $20,000:' THEN 1
    WHEN "column1" = 'Â Â Â Â Â Â Â Â $20,000 to $74,999:' THEN 2
    WHEN "column1" = 'Â Â Â Â Â Â Â Â $75,000 or more:' THEN 3
END;


SELECT 
	--CAST(REPLACE("column4", '%', '') AS FLOAT) AS broadband_percent
	REPLACE("column4", '%', '') AS broadband_percent
INTO broadband_percent_table
FROM CensusData
WHERE "column1" = 'Â Â Â Â Â Â Â Â Â Â Â Â With a broadband Internet subscription';

CREATE TABLE IF NOT EXISTS tablet_series_table (
    tablet TEXT
);

DO $$
DECLARE
    tablet_percent NUMERIC;
BEGIN
    tablet_percent := (SELECT tablet_percentage FROM tablet_percentage_table);

    WITH tablet_series_y AS (
        SELECT 'Yes' AS tablet_y
        FROM generate_series(1, ROUND(tablet_percent * 100))
		)
    INSERT INTO tablet_series_table (tablet)
    SELECT tablet_y
    FROM tablet_series_y;

    WITH tablet_series_n AS (
        SELECT 'No' AS tablet_n
        FROM generate_series(1, (10000 - ROUND(tablet_percent * 100)))
		)
    INSERT INTO tablet_series_table (tablet)
    SELECT tablet_n
    FROM tablet_series_n;
	
END $$;

DROP TABLE IF EXISTS income_bin_series_table;
CREATE TABLE IF NOT EXISTS income_bin_series_table (
    income_bin_series TEXT,
	broadband TEXT
);
DO $$
DECLARE
	income_count_low NUMERIC;
	income_count_mid NUMERIC;
	income_count_high NUMERIC;
	income_count_total NUMERIC;
	broadband_percent_low NUMERIC;
	broadband_percent_mid NUMERIC;
	broadband_percent_high NUMERIC;
BEGIN
	income_count_low := (SELECT income_count FROM income_count_table LIMIT 1);
	income_count_mid := (SELECT income_count FROM income_count_table OFFSET 1 LIMIT 1);
	income_count_high := (SELECT income_count FROM income_count_table OFFSET 2 LIMIT 1);
	income_count_total := (SELECT SUM(income_count) FROM income_count_table);
	broadband_percent_low := (SELECT broadband_percent FROM broadband_percent_table LIMIT 1);
	broadband_percent_mid := (SELECT broadband_percent FROM broadband_percent_table OFFSET 1 LIMIT 1);
	broadband_percent_high := (SELECT broadband_percent FROM broadband_percent_table OFFSET 2 LIMIT 1);
	
	-- LOW
	WITH income_bin_series_low AS (
		SELECT '0-19,999' AS income_bin_low
		FROM generate_series(1, (SELECT ROUND((ROUND(income_count_low/income_count_total, 3)*(broadband_percent_low/100))*10000)))
		)
		
	INSERT INTO income_bin_series_table (income_bin_series)
	SELECT income_bin_low
	FROM income_bin_series_low;
	
	WITH broadband_series_no_low AS (
		SELECT 'No' AS broadband_no_low
		FROM generate_series(1, (SELECT ROUND((ROUND(income_count_low/income_count_total, 3)*10000)-(ROUND(income_count_low/income_count_total, 3)*(broadband_percent_low/100))*10000)))
		)
		
	INSERT INTO income_bin_series_table (broadband)
	SELECT broadband_no_low
	FROM broadband_series_no_low;
	
	UPDATE income_bin_series_table
	SET income_bin_series = COALESCE(income_bin_series, '0-19,999')
	WHERE income_bin_series IS NULL;	
	
	UPDATE income_bin_series_table
	SET broadband = COALESCE(broadband, 'Yes')
	WHERE broadband IS NULL;	
	
	--MID
	WITH income_bin_series_mid AS (
		SELECT '20,000-74,999' AS income_bin_mid
		FROM generate_series(1, (SELECT ROUND((ROUND(income_count_mid/income_count_total, 3)*(broadband_percent_mid/100))*10000)))
		)
		
	INSERT INTO income_bin_series_table (income_bin_series)
	SELECT income_bin_mid
	FROM income_bin_series_mid;
	
	WITH broadband_series_no_mid AS (
		SELECT 'No' AS broadband_no_mid
		FROM generate_series(1, (SELECT ROUND((ROUND(income_count_mid/income_count_total, 3)*10000)-(ROUND(income_count_mid/income_count_total, 3)*(broadband_percent_mid/100))*10000)))
		)
		
	INSERT INTO income_bin_series_table (broadband)
	SELECT broadband_no_mid
	FROM broadband_series_no_mid;
	
	UPDATE income_bin_series_table
	SET income_bin_series = COALESCE(income_bin_series, '20,000-74,999')
	WHERE income_bin_series IS NULL;	
	
	UPDATE income_bin_series_table
	SET broadband = COALESCE(broadband, 'Yes')
	WHERE broadband IS NULL;	
	
	--HIGH
	WITH income_bin_series_high AS (
		SELECT '75,000 and above' AS income_bin_high
		FROM generate_series(1, (SELECT ROUND((ROUND(income_count_high/income_count_total, 3)*(broadband_percent_high/100))*10000)))
		)
		
	INSERT INTO income_bin_series_table (income_bin_series)
	SELECT income_bin_high
	FROM income_bin_series_high;
	
	WITH broadband_series_no_high AS (
		SELECT 'No' AS broadband_no_high
		FROM generate_series(1, (SELECT ROUND((ROUND(income_count_high/income_count_total, 3)*10000)-(ROUND(income_count_high/income_count_total, 3)*(broadband_percent_high/100))*10000)))
		)
		
	INSERT INTO income_bin_series_table (broadband)
	SELECT broadband_no_high
	FROM broadband_series_no_high;
	
	UPDATE income_bin_series_table
	SET income_bin_series = COALESCE(income_bin_series, '75,000 and above')
	WHERE income_bin_series IS NULL;	
	
	UPDATE income_bin_series_table
	SET broadband = COALESCE(broadband, 'Yes')
	WHERE broadband IS NULL;	
		
END $$;

CREATE TABLE new_table (LIKE income_bin_series_table INCLUDING ALL);
INSERT INTO new_table SELECT * FROM income_bin_series_table ORDER BY RANDOM();
DROP TABLE income_bin_series_table;
ALTER TABLE new_table RENAME TO income_bin_series_table;

CREATE TABLE new_table (LIKE tablet_series_table INCLUDING ALL);
INSERT INTO new_table SELECT * FROM tablet_series_table ORDER BY RANDOM();
DROP TABLE tablet_series_table;
ALTER TABLE new_table RENAME TO tablet_series_table;

ALTER TABLE income_bin_series_table 
    ADD COLUMN CaseOrder int
        GENERATED BY DEFAULT AS IDENTITY;
		
ALTER TABLE tablet_series_table 
    ADD COLUMN CaseOrder int
        GENERATED BY DEFAULT AS IDENTITY;


DROP TABLE IF EXISTS final_result_table;

CREATE TABLE final_result_table AS
SELECT income_bin_series_table.income_bin_series, income_bin_series_table.broadband, tablet_series_table.tablet
FROM income_bin_series_table
JOIN tablet_series_table USING (CaseOrder);

ALTER TABLE final_result_table 
    ADD COLUMN "CaseOrder" int
        GENERATED BY DEFAULT AS IDENTITY;
		
ALTER TABLE final_result_table
RENAME COLUMN income_bin_series TO "IncomeBin";
ALTER TABLE final_result_table
RENAME COLUMN broadband TO "Broadband";
ALTER TABLE final_result_table
RENAME COLUMN tablet TO "CensusTablet";

-- \copy final_result_table("CaseOrder", "IncomeBin", "Broadband", "CensusTablet") to 'C:\Users\johnf\OneDrive\Desktop\school\sql\Census.csv' DELIMITER ',' CSV HEADER;