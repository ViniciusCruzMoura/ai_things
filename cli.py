def display_help():
    print("Available commands:")
    print("1. greet - Greets the user")
    print("2. calculate [OPTIONS] - Performs a calculation.")
    print("   Options:")
    print("   --hex     - Display result in hexadecimal")
    print("   --binary  - Display result in binary")
    print("3. exit - Exits the application")

def greet():
    name = input("Enter your name: ")
    print(f"Hello, {name}! Welcome to the CLI app.")

def calculate(options):
    expression = input("Enter a mathematical expression (e.g., 1 + 2): ")
    try:
        result = eval(expression)
        
        if '--hex' in options:
            result = hex(result)
            print(f"Result in hexadecimal: {result}")
        elif '--binary' in options:
            result = bin(result)
            print(f"Result in binary: {result}")
        else:
            print(f"Result: {result}")
    except Exception as e:
        print(f"Error in calculation: {e}")

def main():
    print("Welcome to the CLI Application!")
    display_help()

    while True:
        user_input = input("\nEnter command: ").strip().lower()

        if user_input == "greet":
            greet()
        elif user_input.startswith("calculate"):
            options = user_input.split()[1:]
            calculate(options)
        elif user_input == "exit":
            print("Exiting the application. Goodbye!")
            break
        else:
            print("Unknown command. Type 'help' to see available commands.")

if __name__ == "__main__":
    main()

