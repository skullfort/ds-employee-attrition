-- Exported from QuickDBD: https://www.quickdatabasediagrams.com/
-- NOTE! If you have used non-SQL datatypes in your design, you will have to change these here.

-- Project 4
-- Physical data model

CREATE TABLE enrollee_data (
    enrollee_id int,
    city varchar(255),
    city_development_index float,
    gender varchar(255),
    relevent_experience varchar(255),
    enrolled_university varchar(255),
    education_level varchar(255),
    major_discipline varchar(255),
    experience varchar(255),
    company_size varchar(255),
    company_type varchar(255),
    last_new_job varchar(255),
    training_hours int,
    target float,
    CONSTRAINT pk_enrollee_data PRIMARY KEY (
        enrollee_id
     )
);

