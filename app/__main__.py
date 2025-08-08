import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app", host="0.0.0.0", port=8547, reload=True, reload_dirs=["app"]
    )
