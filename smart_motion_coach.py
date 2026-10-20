
import subprocess
import sys
from pathlib import Path


# ==========================================
# SMART MOTION COACH - MAIN APPLICATION
# ==========================================

# Find the project folder based on this file's location.
# This allows the program to find exercise scripts
# regardless of where the terminal was opened.
PROJECT_FOLDER = Path(__file__).resolve().parent


def run_exercise(filename):
    """Launch an exercise tracker as a separate Python process."""

    file_path = PROJECT_FOLDER / filename

    # Make sure the selected exercise script exists.
    if not file_path.is_file():
        print(f"\nError: {filename} was not found.")
        return

    print(f"\nStarting {filename}...")
    print("Press Q in the camera window to exit.\n")

    try:
        # sys.executable uses the same Python environment
        # that started the main application.
        result = subprocess.run(
            [sys.executable, str(file_path)],
            cwd=PROJECT_FOLDER,
            check=False
        )

        # Report unexpected errors without closing the main menu.
        if result.returncode != 0:
            print(
                f"\n{filename} exited with error code "
                f"{result.returncode}."
            )

    except KeyboardInterrupt:
        print("\nExercise interrupted.")

    print("\nReturned to Smart Motion Coach menu.")


def show_menu():
    """Display the available exercise tracking options."""

    print("\n" + "=" * 45)
    print("           SMART MOTION COACH")
    print("=" * 45)

    print("1. Squat Form Checker")
    print("2. Arm Raise Tracker")
    print("3. Bicep Curl Tracker")
    print("4. Posture Monitor")
    print("5. Exit")

    print("=" * 45)


def main():
    """Keep showing the menu until the user chooses to exit."""

    while True:
        show_menu()

        try:
            choice = input("Select an option (1-5): ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\nExiting Smart Motion Coach.")
            break

        if choice == "1":
            run_exercise("squat_form_checker.py")

        elif choice == "2":
            run_exercise("arm_raise_tracker.py")

        elif choice == "3":
            run_exercise("bicep_curl_tracker.py")

        elif choice == "4":
            run_exercise("posture_monitor.py")

        elif choice == "5":
            print("\nThank you for using Smart Motion Coach!")
            break

        else:
            print("\nInvalid selection. Please choose 1-5.")


if __name__ == "__main__":
    main()
