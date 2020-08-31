# -*- coding: utf-8 -*-

# Importando as bibliotecas necessárias.(Bibliotecas que manipulam o sistema operacional)
import sys, os
import subprocess

# Menu principal
def main_menu():
    os.system('clear')
    
    print("SISTEMA DE RECONHECIMENTO FACIAL - OPENCV\n")
    print("Escolha a tarefa para iniciar:")
    print("1. Registrar novo usuário")
    print("2. Treinar o algoritmo LBPH")
    print("3. Identificacao facial")
    print("\n0. Sair")
    choice = input(" >>  ")
    exec_menu(choice)

    return

# Executa o menu selecionado
def exec_menu(choice):
    os.system('clear')
    ch = choice.lower()
    if ch == '':
        menu_actions['main_menu']()
    else:
        try:
            menu_actions[ch]()
        except KeyError:
            print("Menu inválido, tente novamente.\n")
            menu_actions['main_menu']()
    return

# Menu 1
def menu1():
    print("Entre com os dados para registrar o usuario:\n")
    subprocess.call(["python", "register.py"])
    print("9. Voltar")
    print("0. Sair")
    choice = input(" >>  ")
    exec_menu(choice)
    return


# Menu 2
def menu2():
    print("Treinando o algoritmo...\n")
    subprocess.call(["python", "train.py"])
    print("9. Voltar")
    print("0. Sair" )
    choice = input(" >>  ")
    exec_menu(choice)
    return

# Menu 3
def menu3():
    print("Reconhecimento Facial\n")
    subprocess.call(["python", "recognition.py"])
    print("9. Voltar")
    print("0. Sair")
    choice = input(" >>  ")
    exec_menu(choice)
    return

# Retorna ao menu principal
def back():
    menu_actions['main_menu']()

# Finaliza o programa
def exit():
    sys.exit()

# Opções e menus
menu_actions = {
    'main_menu': main_menu,
    '1': menu1,
    '2': menu2,
    '3': menu3,
    '9': back,
    '0': exit,
}

# Menu Principal
if __name__ == "__main__":
    # Inicializa o Menu Principal
    main_menu()
