-- Exported from QuickDBD: https://www.quickdatabasediagrams.com/
-- NOTE! If you have used non-SQL datatypes in your design, you will have to change these here.

-- Project 4
-- Physical data model

CREATE TABLE enrollee_data (
    enrollee_id int   NOT NULL,
    city varchar(255)   NOT NULL,
    city_development_index float   NOT NULL,
    gender varchar(255)   NOT NULL,
    relevent_experience varchar(255)   NOT NULL,
    enrolled_university varchar(255)   NOT NULL,
    education_level varchar(255)   NOT NULL,
    major_discipline varchar(255)   NOT NULL,
    experience varchar(255)   NOT NULL,
    company_size varchar(255)   NOT NULL,
    company_type varchar(255)   NOT NULL,
    last_new_job varchar(255)   NOT NULL,
    training_hours int   NOT NULL,
    target float   NOT NULL,
    CONSTRAINT pk_enrollee_data PRIMARY KEY (
        enrollee_id
     )
);

