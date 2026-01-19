import uuid

a = uuid.uuid4()
b = f'asdf{a!s}'
print(b)