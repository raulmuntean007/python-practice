a = [1, 2, 3]   # se creează o listă. eticheta a se lipește pe ea
b = a           # NU se creează nimic nou. eticheta b se lipește pe ACEEAȘI listă
c = [1, 2, 3]   # se creează o listă NOUĂ. eticheta c se lipește pe ea

print(id(a))
print(id(b))
print(id(c))
print(a == c)
print(a is c)

a.append(4)

print(a)
print(b)
print(c)



print(a == c)
print(a is c)

#explicatia fiind clara , a is c este False deoarece a și c sunt două obiecte diferite în memorie, chiar dacă conțin aceleași valori. Pe de altă parte, a == c este True deoarece operatorul '==' compară valorile conținute în liste, iar acestea sunt identice.


x = None

print(x is None)
print(x == None)