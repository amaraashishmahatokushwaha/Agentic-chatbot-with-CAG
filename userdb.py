import sqlite3
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def clean_users_db(db_path="data/users.db"):
    """Consolidate redundant users and clean preferences"""
    logger.info("Starting database cleanup...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all users
    cursor.execute("SELECT user_id, name, preferences FROM users")
    users = cursor.fetchall()

    # Group users by name
    user_groups = {}
    for user_id, name, preferences in users:
        if name not in user_groups:
            user_groups[name] = []
        user_groups[name].append({"user_id": user_id, "preferences": preferences})

    # Consolidate preferences for each name
    for name, user_list in user_groups.items():
        if len(user_list) == 1:
            logger.info(f"No duplicates for user '{name}', skipping consolidation")
            continue

        logger.info(f"Consolidating {len(user_list)} entries for user '{name}'...")
        # Keep the first user_id with non-empty preferences, or the first one
        main_user = None
        for user in user_list:
            try:
                prefs = json.loads(user["preferences"]) if user["preferences"] else {}
                if prefs:
                    main_user = user
                    break
            except json.JSONDecodeError:
                logger.warning(f"Invalid preferences JSON for user_id {user['user_id']}")
        if not main_user:
            main_user = user_list[0]

        # Merge preferences
        merged_prefs = {}
        for user in user_list:
            try:
                prefs = json.loads(user["preferences"]) if user["preferences"] else {}
                for category, items in prefs.items():
                    if category not in merged_prefs:
                        merged_prefs[category] = []
                    for item in items:
                        if item not in merged_prefs[category]:
                            merged_prefs[category].append(item)
            except json.JSONDecodeError:
                logger.warning(f"Invalid preferences JSON for user_id {user['user_id']}")

        # Remove verbs from preferences
        if "activities" in merged_prefs:
            logger.info(f"Removing verbs from activities for user '{name}'")
            del merged_prefs["activities"]

        # Update main user
        main_user_id = main_user["user_id"]
        updated_prefs_json = json.dumps(merged_prefs)
        cursor.execute(
            "UPDATE users SET preferences = ? WHERE user_id = ?",
            (updated_prefs_json, main_user_id)
        )
        logger.info(f"Updated preferences for user '{name}' (user_id: {main_user_id}): {merged_prefs}")

        # Delete redundant users
        for user in user_list:
            if user["user_id"] != main_user_id:
                cursor.execute("DELETE FROM users WHERE user_id = ?", (user["user_id"],))
                logger.info(f"Deleted redundant user_id: {user['user_id']} for name '{name}'")

    conn.commit()
    conn.close()
    logger.info("Database cleanup completed")

def check_users_db(db_path="data/users.db"):
    """Check the contents of the users SQLite database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT user_id, name, preferences, created_at FROM users")
    users = cursor.fetchall()

    if not users:
        logger.info("No users found in the database.")
        conn.close()
        return

    logger.info(f"Found {len(users)} user(s):")
    for user in users:
        user_id, name, preferences, created_at = user
        try:
            prefs = json.loads(preferences) if preferences else {}
        except json.JSONDecodeError:
            logger.warning(f"Invalid preferences JSON for user {name}: {preferences}")
            prefs = {}
        logger.info(f"User ID: {user_id}")
        logger.info(f"Name: {name}")
        logger.info(f"Preferences: {prefs}")
        logger.info(f"Created At: {created_at}")
        logger.info("-" * 50)
    conn.close()

if __name__ == "__main__":
    clean_users_db()
    logger.info("Final database state:")
    check_users_db()