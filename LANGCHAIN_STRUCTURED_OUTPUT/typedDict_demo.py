from typing import TypedDict

#definition of dictionary
class Person(TypedDict):
    name:str
    age:int

new_person: Person ={'name':'Aeshah','age':20}

print(new_person)