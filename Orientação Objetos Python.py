class Quadrado: 
    '''Classe para calcular o quadrado de um número'''
    def __init__(self, valor): #metodo construtor da classe
        self.x = valor
        print('Objeto criado!')
    
    def calcula_quadrado(self):
        Quadrado = self.x * self.x 
        return (f'Cubo calculado: {Quadrado}')
    
print(type(Quadrado))

teste = Quadrado(6)
c = teste.calcula_quadrado()
print(c)