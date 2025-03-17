import mysql.connector
from mysql.connector import Error
import numpy as np
import json
import hashlib

class DatabaseConnector:
    def __init__(self, host="localhost", user="root", password="your_password", database="your_database"):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
        self.connect()
        self.create_tables()

    def connect(self):
        """建立数据库连接"""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            if self.connection.is_connected():
                print("Successfully connected to MySQL database")
        except Error as e:
            print(f"Error while connecting to MySQL: {e}")

    def close(self):
        """关闭数据库连接"""
        if self.connection.is_connected():
            self.connection.close()
            print("MySQL connection is closed")


    # =============================================================================
    # 创建数据表
    def create_tables(self):
        """创建数据表"""
        cursor = self.connection.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS `action_def` (
            `id` INT PRIMARY KEY,
            `name` TEXT
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS `action_dict` (
            `id` INT AUTO_INCREMENT PRIMARY KEY,
            `action_tensor` JSON,
            `hash_value` CHAR(32) UNIQUE
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS `action_log` (
            `id` INT AUTO_INCREMENT PRIMARY KEY,
            `action_log` JSON
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS `state_dict` (
            `id` INT AUTO_INCREMENT PRIMARY KEY,
            `allies` JSON,
            `enemies` JSON,
            `hp_amount` INT,
            `hp_mean` FLOAT,
            `hp_variance` FLOAT,
            `hash_value` CHAR(32) UNIQUE
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS `state_log` (
            `id` INT AUTO_INCREMENT PRIMARY KEY,
            `state_log` JSON
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS `im_state_dict` (
            `id` INT AUTO_INCREMENT PRIMARY KEY,
            `id_check` INT,
            `hash_code` VARCHAR(64) UNIQUE,
            `hp_amount` INT,
            `hp_mean` FLOAT,
            `hp_variance` FLOAT
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS `im_state_log` (
            `id` INT AUTO_INCREMENT PRIMARY KEY,
            `im_state_log` JSON
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS `state_map` (
            `id` INT AUTO_INCREMENT PRIMARY KEY,
            `im_state` VARCHAR(64) UNIQUE,
            `state` JSON
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS `result_log` (
            `id` INT AUTO_INCREMENT PRIMARY KEY,
            `mode` VARCHAR(64),
            `result` VARCHAR(64),
            `reward` FLOAT
        )
        """)

    # =============================================================================
    # 插入数据
    def insert_action_def(self):
        """插入 action_def 数据"""
        actions = {
            0: "no-op",
            1: "stop",
            2: "move north",
            3: "move south",
            4: "move east",
            5: "move west",
            6: "attack 0",
            7: "attack 1",
            8: "attack 2",
            9: "attack 3"
        }
        cursor = self.connection.cursor()
        action_data = [(key, value) for key, value in actions.items()]
        insert_query = """
        INSERT INTO `action_def` (`id`, `name`) VALUES (%s, %s)
        """
        cursor.executemany(insert_query, action_data)
        self.connection.commit()
        cursor.close()

    def insert_action_dict(self, action_dict):
        """批量插入 action_dict 数据"""
        cursor = self.connection.cursor()
        for key, value in action_dict.items():
            action_tensor_json = json.dumps(value, sort_keys=True)
            hash_value = hashlib.md5(action_tensor_json.encode()).hexdigest()  # 计算哈希值
            cursor.execute("""
                SELECT COUNT(*) FROM `action_dict` WHERE `hash_value` = %s
                """, (hash_value,))
            result = cursor.fetchone()
            if result[0] == 0:
                cursor.execute("""
                    INSERT INTO `action_dict` (`action_tensor`, `hash_value`)
                    VALUES (%s, %s)
                    """, (action_tensor_json, hash_value))
            #     print(f"Inserted: {value}")
            # else:
            #     print(f"Skipped (already exists): {value}")
        self.connection.commit()
        cursor.close()

    def insert_action_log(self, action_log):
        """批量插入 action_log 数据"""
        cursor = self.connection.cursor()
        insert_query = """
        INSERT INTO `action_log` (`action_log`) VALUES (%s)
        """
        cursor.executemany(insert_query, [(json.dumps(log),) for log in action_log])
        self.connection.commit()
        cursor.close()

    def insert_state_dict(self, state_dict):
        """批量插入 state_dict 数据"""
        cursor = self.connection.cursor()
        for key, value in state_dict.items():
            allies = value[0]["allies"]
            enemies = value[0]["enemies"]
            hp_amount = value[1]
            hp_mean = value[2]
            hp_variance = value[3]

            allies_json = json.dumps(allies, sort_keys=True)
            enemies_json = json.dumps(enemies, sort_keys=True)

            combined_string = allies_json + enemies_json
            hash_value = hashlib.md5(combined_string.encode()).hexdigest()

            cursor.execute("""
                SELECT `id`, `hp_amount`, `hp_mean`, `hp_variance` 
                FROM `state_dict` 
                WHERE `hash_value` = %s
                """, (hash_value,))
            result = cursor.fetchone()
            if result:
                record_id, old_hp_amount, old_hp_mean, old_hp_variance = result
                new_hp_amount = old_hp_amount + hp_amount

                new_hp_mean = (old_hp_mean * old_hp_amount + hp_mean * hp_amount) / new_hp_amount

                new_hp_variance = (old_hp_amount * (old_hp_variance + (old_hp_mean - new_hp_mean) ** 2) +
                                   hp_amount * (hp_variance + (hp_mean - new_hp_mean) ** 2)) / new_hp_amount

                cursor.execute("""
                        UPDATE `state_dict`
                        SET `hp_amount` = %s, `hp_mean` = %s, `hp_variance` = %s
                        WHERE `id` = %s
                        """, (new_hp_amount, new_hp_mean, new_hp_variance, record_id))
                # print(f"Updated: {key} -> HP Amount: {new_hp_amount}, HP Mean: {new_hp_mean}, HP Variance: {new_hp_variance}")
            else:
                cursor.execute("""
                        INSERT INTO `state_dict` (`allies`, `enemies`, `hp_amount`, `hp_mean`, `hp_variance`, `hash_value`)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """, (allies_json, enemies_json, hp_amount, hp_mean, hp_variance, hash_value))
                # print(f"Inserted: {key} -> Allies: {allies}, Enemies: {enemies}, HP Amount: {hp_amount}, HP Mean: {hp_mean}, HP Variance: {hp_variance}")
        self.connection.commit()
        cursor.close()

    def insert_state_log(self, state_log):
        """批量插入 state_log 数据"""
        cursor = self.connection.cursor()
        insert_query = """
        INSERT INTO `state_log` (`state_log`) VALUES (%s)
        """
        cursor.executemany(insert_query, [(json.dumps(log),) for log in state_log])
        self.connection.commit()
        cursor.close()

    def insert_im_state_dict(self, im_state_dict):
        """批量插入 im_state_dict 数据"""
        cursor = self.connection.cursor()
        for hash_code, values in im_state_dict.items():
            id_check, hp_amount, hp_mean, hp_variance = values

            # 检查 hash_code 是否已存在
            cursor.execute("""
            SELECT `id`, `id_check`, `hp_amount`, `hp_mean`, `hp_variance` 
            FROM `im_state_dict` 
            WHERE `hash_code` = %s
            """, (hash_code,))
            result = cursor.fetchone()

            if result:  # 如果已存在，更新记录
                record_id, id_check, old_hp_amount, old_hp_mean, old_hp_variance = result
                new_hp_amount = old_hp_amount + hp_amount

                new_hp_mean = (old_hp_mean * old_hp_amount + hp_mean * hp_amount) / new_hp_amount

                new_hp_variance = (old_hp_amount * (old_hp_variance + (old_hp_mean - new_hp_mean) ** 2) +
                                   hp_amount * (hp_variance + (hp_mean - new_hp_mean) ** 2)) / new_hp_amount

                cursor.execute("""
                UPDATE `im_state_dict`
                SET `hp_amount` = %s, `hp_mean` = %s, `hp_variance` = %s
                WHERE `id` = %s
                """, (new_hp_amount, new_hp_mean, new_hp_variance, record_id))
                # print(f"Updated: {hash_code} -> HP Amount: {new_hp_amount}, HP Mean: {new_hp_mean}, HP Variance: {new_hp_variance}")
            else:  # 如果不存在，插入新记录
                cursor.execute("""
                INSERT INTO `im_state_dict` (`id_check`, `hash_code`, `hp_amount`, `hp_mean`, `hp_variance`)
                VALUES (%s, %s, %s, %s, %s)
                """, (id_check, hash_code, hp_amount, hp_mean, hp_variance))
                # print(f"Inserted: {hash_code} -> HP Amount: {hp_amount}, HP Mean: {hp_mean}, HP Variance: {hp_variance}")
        self.connection.commit()
        cursor.close()

    def insert_im_state_log(self, im_state_log):
        """批量插入 im_state_log 数据"""
        cursor = self.connection.cursor()
        insert_query = """
        INSERT INTO `im_state_log` (`im_state_log`) VALUES (%s)
        """
        cursor.executemany(insert_query, [(json.dumps(log),) for log in im_state_log])
        self.connection.commit()
        cursor.close()

    def insert_state_map(self, state_map):
        """批量插入 state_map 数据"""
        cursor = self.connection.cursor()
        for im_state, new_state_data in state_map.items():
            # 将新数据转换为 JSON 字符串
            new_state_json = json.dumps(new_state_data)

            # 检查 im_state 是否已存在
            cursor.execute("""
            SELECT `id`, `state` 
            FROM `state_map` 
            WHERE `im_state` = %s
            """, (im_state,))
            result = cursor.fetchone()

            if result:  # 如果已存在，合并数据并更新记录
                record_id, old_state_json = result
                old_state_data = json.loads(old_state_json)  # 将 JSON 字符串转换为列表
                new_unique_elements = [item for item in new_state_data if item not in old_state_data]
                if not new_unique_elements:
                    pass
                    # print(f"No new elements to insert for {im_state}. Existing state: {old_state_data}")
                else:
                    # 合并新旧数据
                    combined_state_data = old_state_data + new_unique_elements
                    combined_state_json = json.dumps(combined_state_data)  # 将合并后的列表转换为 JSON 字符串
                    cursor.execute("""
                                    UPDATE `state_map`
                                    SET `state` = %s
                                    WHERE `id` = %s
                                    """, (combined_state_json, record_id))
                    # print(f"Updated: {im_state} -> State: {combined_state_data}")

            else:  # 如果不存在，插入新记录
                cursor.execute("""
                INSERT INTO `state_map` (`im_state`, `state`)
                VALUES (%s, %s)
                """, (im_state, new_state_json))
                # print(f"Inserted: {im_state} -> State: {new_state_data}")
        self.connection.commit()
        cursor.close()

    def insert_result_log(self, result_log):
        cursor = self.connection.cursor()
        insert_query = """
        INSERT INTO `result_log` (`mode`, `result`, `reward`)
        VALUES (%s, %s, %s)
        """
        cursor.executemany(insert_query, [(log[0], log[1], log[2]) for log in result_log])
        self.connection.commit()
        cursor.close()

    # =============================================================================
    # 查询数据
    def select_action_dict(self):
        """查询 action_dict 数据"""
        cursor = self.connection.cursor()
        cursor.execute("SELECT `id`, `action_tensor`, `hash_value` FROM `action_dict`")
        data = cursor.fetchall()

        restored_action_dict = {}
        for row in data:
            tensor_id = row[0]
            restored_action_dict[tensor_id] = {
                "action_tensor": np.array(json.loads(row[1])),
                "hash_value": row[2]
            }
        cursor.close()
        return restored_action_dict

    def select_action_log(self):
        cursor = self.connection.cursor()
        cursor.execute("SELECT `id`, `action_log` FROM `action_log`")
        results = cursor.fetchall()
        restored_action_logs = {row[0]: json.loads(row[1]) for row in results}
        cursor.close()
        return restored_action_logs

    def select_state_dict(self):
        cursor = self.connection.cursor()
        cursor.execute("SELECT `id`, `allies`, `enemies`, `hp_amount`, `hp_mean`, `hp_variance`, `hash_value` FROM `state_dict`")
        results = cursor.fetchall()
        restored_state_dict = {}
        for row in results:
            state_id = row[0]
            restored_state_dict[state_id] = {
                "allies": np.array(json.loads(row[1])),
                "enemies": np.array(json.loads(row[2])),
                "hp_amount": row[3],
                "hp_mean": row[4],
                "hp_variance": row[5],
                "hash_value": row[6]
            }
        cursor.close()
        return restored_state_dict

    def select_state_log(self):
        cursor = self.connection.cursor()
        cursor.execute("SELECT `id`, `state_log` FROM `state_log`")
        results = cursor.fetchall()
        restored_state_logs = {row[0]: json.loads(row[1]) for row in results}
        cursor.close()
        return restored_state_logs

    def select_im_state_dict(self):
        cursor = self.connection.cursor()
        cursor.execute("SELECT `id`, `hash_code`, `hp_amount`, `hp_mean`, `hp_variance` FROM `im_state_dict`")
        results = cursor.fetchall()

        restored_im_state_dict = {}
        for row in results:
            state_id = row[0]
            restored_im_state_dict[state_id] = {
                "hash_code": row[1],
                "hp_amount": row[2],
                "hp_mean": row[3],
                "hp_variance": row[4]
            }
        cursor.close()
        return restored_im_state_dict

    def select_im_state_log(self):
        cursor = self.connection.cursor()
        cursor.execute("SELECT `id`, `im_state_log` FROM `im_state_log`")
        results = cursor.fetchall()

        restored_im_state_logs = {row[0]: json.loads(row[1]) for row in results}
        cursor.close()
        # print("Restored Immediate State Logs:")
        # for log_id, log in restored_im_state_logs.items():
        #     print(f"ID: {log_id}, Log: {log}")
        return restored_im_state_logs

    def select_state_map(self):
        cursor = self.connection.cursor()
        cursor.execute("SELECT `id`, `im_state`, `state` FROM `state_map`")
        results = cursor.fetchall()

        restored_state_map = {}
        for row in results:
            im_state = row[1]
            state = json.loads(row[2])
            restored_state_map[im_state] = state
        # print("Restored State Map:")
        # for key, value in restored_state_map.items():
        #     print(f"Immediate State: {key}, State: {value}")

        cursor.close()
        return restored_state_map

    def select_result_log(self):
        cursor = self.connection.cursor()
        cursor.execute("SELECT `id`, `mode`, `result`, `reward` FROM `result_log`")
        results = cursor.fetchall()

        restored_result_logs =  {row[0]: [row[1], row[2], row[3]] for row in results}
        cursor.close()
        return restored_result_logs

    def select_processed_action_dict(self):
        """查询 processed_action_dict 数据"""
        cursor = self.connection.cursor()
        cursor.execute("SELECT `id`, `action_tensor`, `hash_value` FROM `processed_action_dict`")
        data = cursor.fetchall()

        restored_action_dict = {}
        for row in data:
            tensor_id = row[0]
            restored_action_dict[tensor_id] = {
                "action_tensor": np.array(json.loads(row[1])),
                "hash_value": row[2]
            }
        cursor.close()
        return restored_action_dict

    def select_processed_action_log(self):
        cursor = self.connection.cursor()
        cursor.execute("SELECT `id`, `action_log` FROM `processed_action_log`")
        results = cursor.fetchall()
        restored_action_logs = {row[0]: json.loads(row[1]) for row in results}
        cursor.close()
        return restored_action_logs

    def select_processed_state_dict(self):
        cursor = self.connection.cursor()
        cursor.execute("SELECT `id`, `allies`, `enemies`, `hp_amount`, `hp_mean`, `hp_variance` FROM `processed_state_dict`")
        results = cursor.fetchall()
        restored_state_dict = {}
        for row in results:
            state_id = row[0]
            restored_state_dict[state_id] = {
                "allies": np.array(json.loads(row[1])),
                "enemies": np.array(json.loads(row[2])),
                "hp_amount": row[3],
                "hp_mean": row[4],
                "hp_variance": row[5]
            }
        cursor.close()
        return restored_state_dict

    def select_processed_state_log(self):
        cursor = self.connection.cursor()
        cursor.execute("SELECT `id`, `state_log` FROM `processed_state_log`")
        results = cursor.fetchall()
        restored_state_logs = {row[0]: json.loads(row[1]) for row in results}
        cursor.close()
        return restored_state_logs

    def select_processed_state_map(self):
        cursor = self.connection.cursor()
        cursor.execute("SELECT `id`, `precessed_state`, `state` FROM `processed_state_map`")
        results = cursor.fetchall()

        restored_state_map = {}
        for row in results:
            im_state = row[1]
            state = json.loads(row[2])
            restored_state_map[im_state] = state

        cursor.close()
        return restored_state_map

    # =============================================================================
    # 高级操作
    def send_data_to_database(self, s_dict, s_log, a_dict, a_log, s_map):
        cursor = self.connection.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS `processed_action_dict` (
            `id` INT AUTO_INCREMENT PRIMARY KEY,
            `action_tensor` JSON,
            `hash_value` CHAR(32) UNIQUE
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS `processed_action_log` (
            `id` INT AUTO_INCREMENT PRIMARY KEY,
            `action_log` JSON
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS `processed_state_dict` (
            `id` INT AUTO_INCREMENT PRIMARY KEY,
            `allies` JSON,
            `enemies` JSON,
            `hp_amount` INT,
            `hp_mean` FLOAT,
            `hp_variance` FLOAT
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS `processed_state_log` (
            `id` INT AUTO_INCREMENT PRIMARY KEY,
            `state_log` JSON
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS `processed_state_map` (
            `id` INT AUTO_INCREMENT PRIMARY KEY,
            `precessed_state` VARCHAR(64) UNIQUE,
            `state` JSON
        )
        """)

        for key, value in a_dict.items():
            action_tensor_json = json.dumps(value['action_tensor'].tolist(), sort_keys=True)
            hash_value = value['hash_value']
            cursor.execute("""
                INSERT INTO `processed_action_dict` (`action_tensor`, `hash_value`)
                VALUES (%s, %s)
                """, (action_tensor_json, hash_value))

        insert_query = """
        INSERT INTO `processed_action_log` (`action_log`) VALUES (%s)
        """
        cursor.executemany(insert_query, [(json.dumps(log),) for log in a_log.values()])

        for key, value in s_dict.items():
            allies = value["allies"]
            enemies = value["enemies"]
            hp_amount = value["hp_amount"]
            hp_mean = value["hp_mean"]
            hp_variance = value["hp_variance"]

            allies_json = json.dumps(allies.tolist(), sort_keys=True)
            enemies_json = json.dumps(enemies.tolist(), sort_keys=True)

            cursor.execute("""
                INSERT INTO `processed_state_dict` (`allies`, `enemies`, `hp_amount`, `hp_mean`, `hp_variance`)
                VALUES (%s, %s, %s, %s, %s)
                """, (allies_json, enemies_json, hp_amount, hp_mean, hp_variance))

        insert_query = """
        INSERT INTO `processed_state_log` (`state_log`) VALUES (%s)
        """
        cursor.executemany(insert_query, [(json.dumps(log),) for log in s_log.values()])

        for pro_state, new_state_data in s_map.items():
            new_state_json = json.dumps(list(new_state_data), sort_keys=True)
            cursor.execute("""
                INSERT INTO `processed_state_map` (`precessed_state`, `state`)
                VALUES (%s, %s)
                """, (pro_state, new_state_json))

        self.connection.commit()
        cursor.close()
