
from pydantic import BaseModel,EmailStr,Field
from typing import Optional

class Student(BaseModel):
    name:str='Aeshah'
    age:Optional[int]=None
    email:EmailStr
    cgpa:float=Field(gt=0,lt=10,default=5,description='A decimal value represennting cgpa of student')

new_student={'age':20,'email':'abc@gmail.com'}

student=Student(**new_student)

student_dict=dict(student)
student_json=student.model_dump_json()
print(student_json)