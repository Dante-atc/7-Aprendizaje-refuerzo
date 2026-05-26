

"""
El problema del blackjack simplificado como un problema de aprendizaje por refuerzo
"""


from RL import MDPsim,SARSA,Q_learning,PoliticaGreedy
from random import randint

class BlackJack(MDPsim):
    
    def __init__(self,gama):
        self.estados=[(j,d,a) for j in range(12,22) for d in range(1,11) for a in [False,True]]+[("T",0,False)]
        self.gama=gama; self.terminal=False; self.resultado=None
    
    def reparte_carta(self): return min(randint(1,13),10)
    
    def evalua(self,c):
        s=sum(c)
        return (s+10,True) if 1 in c and s+10<=21 else (s,False)
   
    def estado_inicial(self):
        while True:
            self.terminal=False; self.resultado=None
            self.mano_jugador=[self.reparte_carta(),self.reparte_carta()]
            self.mano_crupier=[self.reparte_carta(),self.reparte_carta()]
            self.carta_visible=self.mano_crupier[0]
            sm,_=self.evalua(self.mano_jugador)

            while sm<12: self.mano_jugador+=[self.reparte_carta()]; sm,_=self.evalua(self.mano_jugador)
            self.suma_jugador,self.as_usable=self.evalua(self.mano_jugador)
            jb=self.suma_jugador==21; cb=self.evalua(self.mano_crupier)[0]==21

            if jb or cb:
                self.terminal=True
                self.resultado="empate" if jb and cb else "blackjack" if jb else "pierde"

            if not self.terminal:
                return (self.suma_jugador,self.carta_visible,self.as_usable)
    
    def acciones_legales(self,s): return [] if self.es_terminal(s) else [0,1]
    
    def recompensa(self,s,a,s_): return {"blackjack":1.5,"gana":1,"empate":0,"pierde":-1,"bust":-1}.get(self.resultado,0)
    
    
    def sucesor(self,s,a):
        if a==1:
            self.mano_jugador+=[self.reparte_carta()]
            self.suma_jugador,self.as_usable=self.evalua(self.mano_jugador)

            if self.suma_jugador>21:
                self.terminal=True
                self.resultado="bust"

        else:
            self.terminal=True

            while self.evalclearua(self.mano_crupier)[0]<17:
                self.mano_crupier+=[self.reparte_carta()]

            dc=self.evalua(self.mano_crupier)[0]

            self.resultado=(
                "gana" if dc>21 or self.suma_jugador>dc
                else "pierde" if self.suma_jugador<dc
                else "empate"
            )

        return ("T",0,False) if self.terminal else (
            self.suma_jugador,
            self.carta_visible,
            self.as_usable
        )

    def transicion(self,s,a):
        return self.sucesor(s,a)
    
    def es_terminal(self,s): return s==("T",0,False)


if __name__=="__main__":
    blackjack=BlackJack(gama=1)

    Q_sarsa=SARSA(blackjack,alfa=0.1,epsilon=0.1,n_ep=5000,n_iter=100)

    Q_learning=Q_learning(blackjack,alfa=0.1,epsilon=0.1,n_ep=5000,n_iter=100)

    pi_s=PoliticaGreedy(Q_sarsa); pi_q=PoliticaGreedy(Q_learning)

    print("Estado".center(10)+'|'+"SARSA".center(10)+'|'+"Q-learning".center(10))

    for s in blackjack.estados:
        if not blackjack.es_terminal(s): print(str(s).center(10)+'|'+str(pi_s(s)).center(10)+'|'+str(pi_q(s)).center(10))

"""
****************************************************************************************
Responde las siguientes preguntas:

1. ¿Cuáles son los estados, acciones, recompensas y transiciones en el problema del 
    blackjack?  
    Estados=(jugador,dealer,as); acciones=0/1; recompensas finales; transiciones=cartas

2. ¿Cómo se pueden representar los estados del blackjack de manera eficiente para el 
    aprendizaje por refuerzo?
    Tupla(suma,dealer,asusable)

3. ¿Qué pasa si se modifica el valor de epsilón de la política epsilon-greedy?
    Un epsilon alto es igual a mas exploracion

4. ¿Cómo afecta el valor de alfa en la convergencia de los algoritmos SARSA y Q-learning?
    Un alfa alto hace que aprenda rapido pero menos estable

5. ¿Cuál de los dos algoritmos, SARSA o Q-learning, consideras que es más adecuado para 
   el problema del blackjack y por qué?
    Q-learning es más adecuado para Blackjack porque aprende directamente la 
    política óptima (off-policy), sin depender de las acciones exploratorias,
    permitiendo encontrar estrategias más eficientes que SARSA.
   
6. ¿Se puede explicar con cierta lógica del juego la política óptima encontrada por cada 
   algoritmo? ¿Qué acciones se toman en cada estado y por qué?
   Sí. La política óptima sigue la lógica del Blackjack: pedir (Hit) cuando la 
   suma es baja y hay poco riesgo de pasarse, y plantarse (Stand) cuando la suma
    es alta para evitar perder. También influye la carta visible del crupier; 
    si el crupier muestra una carta fuerte, el jugador suele tomar más riesgos.
****************************************************************************************
"""
