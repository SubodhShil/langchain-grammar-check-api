from grammar_check_api import Application

# Create the application instance
app_instance = Application()

# Get the FastAPI app
app = app_instance.get_app()

# This allows the app to be run with uvicorn directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)