## Setup Instructions

Follow these steps to get started with the Prayer Attendance System:

1. **Install Conda**: Ensure you have Conda installed on your system.
2. **Create Environment**: Run the following command to create the environment from the `environment.yml` file:
   ```sh
   conda env create -f environment.yml
   ```
3. **Activate Environment**: Activate the newly created environment:
   ```sh
   conda activate attandance
   ```
4. **Check Python Activation**: Verify that Python is active in the environment:
   ```sh
   which python
   ```
5. **Fetch and Save API Data**: Execute the script to fetch and save the necessary API data:
   ```sh
   python fetch_and_save_api_data.py
   ```

You're all set! Enjoy using the Prayer Attendance System.
