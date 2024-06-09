
def create_student_dict(student):
    return {
        'student_id': student[0],
        'student_code': student[1],
        'first_name': student[2],
        'last_name': student[3],
        'birth_date': student[4],
        'enrollment_date': student[5],
        'major': student[6]
    }
def create_student_attendance_dict(student_attendance):
    return {
        'student_id': student_attendance[0],
        'student_code': student_attendance[1],
        'first_name': student_attendance[2],
        'last_name': student_attendance[3],
        'birth_date': student_attendance[4],
        'enrollment_date': student_attendance[5],
        'major': student_attendance[6],
        'attendance_id': student_attendance[7],
        'attendance_date': student_attendance[8],
        'status': student_attendance[9]
    }