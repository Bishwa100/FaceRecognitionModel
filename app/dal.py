from flask import request, jsonify, Blueprint
from .utils import process_frame
from .config import get_db_connection

def insert_student(student):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO student_data.student (student_id, first_name, last_name, birth_date, enrollment_date, major)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (student['student_id'], student['first_name'], student['last_name'], student['birth_date'], student['enrollment_date'], student['major']))
        conn.commit()
        cur.close()
        return True
    except Exception as e:
        conn.rollback()
        print(f"Error: {e}")
        return False

def get_all_students():
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT student_id, student_code, first_name, last_name, birth_date, enrollment_date, major 
                FROM student_data.student
            """)
            students = cur.fetchall()
        conn.close()
        return (True, students)
    except Exception as e:
        print(f"Error: {e}")
        return (False, str(e))


def get_student_attendance(student_code, attendance_date):
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    s.student_id,
                    s.student_code,
                    s.first_name,
                    s.last_name,
                    s.birth_date,
                    s.enrollment_date,
                    s.major,
                    a.attendance_id,
                    a.attendance_date,
                    a.status
                FROM
                    student_data.student s
                JOIN
                    student_data.attendance a
                ON
                    s.student_code = a.student_code
                WHERE
                    a.attendance_date = %s
                    AND s.student_code = %s
            """, (attendance_date, student_code))
            records = cur.fetchall()
        conn.close()
        return (True, records)
    except Exception as e:
        print(f"Error: {e}")
        return (False, str(e))


