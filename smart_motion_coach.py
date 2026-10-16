
import subprocess
import sys
from pathlib import Path

# Smart Motion Coach main application
PROJECT_FOLDER = Path(__file__).resolve().parent


def run_exercise(filename):
    """Launch an exercise tracking program."""

    file_path = PROJECT_FOLDER / filename

    if not file_path.exists():
        print(f"\nError: {filename} does not exist yet.")
        return

    print(f"\nStarting {filename}...")
    print("Press Q in the camera window to exit.\n")

    try:
        subprocess.run(
            [sys.executable, str(file_path)],
            cwd=PROJECT_FOLDER,
            check=False
        )
    except KeyboardInterrupt:
        print("\nExercise stopped.")

    print("\nReturned to Smart Motion Coach menu.")


def show_menu():
    """Display the available exercise options."""

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
    """Run the main application menu."""

    while True:
        show_menu()

        choice = input("Select an option (1-5): ").strip()

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
