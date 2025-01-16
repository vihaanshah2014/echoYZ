import argparse

class TodoList:
    def __init__(self):
        self.tasks = []

    def add_task(self, description):
        self.tasks.append({'description': description, 'completed': False})
        print(f"Added task: {description}")

    def remove_task(self, index):
        if 0 <= index < len(self.tasks):
            removed_task = self.tasks.pop(index)
            print(f"Removed task: {removed_task['description']}")
        else:
            print(f"Task index {index} is out of range.")

    def view_tasks(self):
        if not self.tasks:
            print("No tasks available.")
        for i, task in enumerate(self.tasks):
            status = "Done" if task['completed'] else "Pending"
            print(f"{i}: {task['description']} [{status}]")

    def complete_task(self, index):
        if 0 <= index < len(self.tasks):
            self.tasks[index]['completed'] = True
            print(f"Marked task as completed: {self.tasks[index]['description']}")
        else:
            print(f"Task index {index} is out of range.")

def main():
    parser = argparse.ArgumentParser(description="Simple To-Do List Manager")
    parser.add_argument('command', choices=['add', 'remove', 'view', 'complete'])
    parser.add_argument('arguments', nargs='*', help="Arguments for the command")

    args = parser.parse_args()

    todo = TodoList()

    if args.command == 'add':
        todo.add_task(" ".join(args.arguments))
    elif args.command == 'remove':
        todo.remove_task(int(args.arguments[0]))
    elif args.command == 'view':
        todo.view_tasks()
    elif args.command == 'complete':
        todo.complete_task(int(args.arguments[0]))

if __name__ == "__main__":
    main()