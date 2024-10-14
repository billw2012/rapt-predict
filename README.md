# RAPT Predict

Reads your Pill Hydrometer data from the specified date (edit in the source for now), and uses recent history of SG values to predict FG. Shows the results in an interactive graph.

# Usage

Create a file in the same directory called `credentials.json` of the form
```json
{
    "username": "user registered email",
    "api_key": "API Key"
}
```

Edit the `start_date` in the source on line 165, then run.

Example output:

![image](https://github.com/user-attachments/assets/81657959-e0f2-4c72-8828-4c64fe2bb2ab)
