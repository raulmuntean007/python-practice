lista_a = [1, 2]
lista_b = lista_a
lista_a.append(3)

print(lista_a)
print(lista_b)

text_a = "ab"
text_b = text_a
text_a = text_a + "c"

print(text_a)
print(text_b)

lista_c = [1, 2]
print(id(lista_c))
lista_c.append(3)
print(id(lista_c))

text_c = "ab"
print(id(text_c))
text_c = text_c + "c"
print(id(text_c)) 

lista_d = [1, 2]
print(id(lista_d))
lista_d = lista_d + [3]
print(id(lista_d))


print("de aici")

lista_e = [1, 2]
lista_f = lista_e
lista_e += [3]

print(lista_f)