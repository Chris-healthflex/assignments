from app.database.mongodb import MongoDB


def main():
    print("Connecting to MongoDB...")

    database = MongoDB()

    database.client.admin.command("ping")

    print("MongoDB connection successful")


if __name__ == "__main__":
    main()