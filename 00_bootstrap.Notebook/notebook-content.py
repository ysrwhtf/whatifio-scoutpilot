# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "231954fd-0578-472d-894d-7905cdf13a6b",
# META       "default_lakehouse_name": "scout_lh",
# META       "default_lakehouse_workspace_id": "f3a9deba-3b22-4ae2-95c0-27c1d12e8f49",
# META       "known_lakehouses": [
# META         {
# META           "id": "231954fd-0578-472d-894d-7905cdf13a6b"
# META         }
# META       ]
# META     }
# META   }
# META }

# CELL ********************

from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    DoubleType, TimestampType, DateType
)

def create(schema, name, cols):
    fqn = f"{schema}.{name}"
    spark.createDataFrame([], StructType(cols)) \
        .write.mode("ignore").saveAsTable(fqn)
    print(f"ok: {fqn}")

# --- BRONZE (raw landings) ---
create("bronze", "detections", [
    StructField("session_id", StringType()),
    StructField("frame",      IntegerType()),
    StructField("track_id",   IntegerType()),
    StructField("bbox_x",     DoubleType()),
    StructField("bbox_y",     DoubleType()),
    StructField("bbox_w",     DoubleType()),
    StructField("bbox_h",     DoubleType()),
    StructField("confidence", DoubleType()),
])

# --- SILVER (cleaned, court coords) ---
create("silver", "sessions", [
    StructField("session_id", StringType()),
    StructField("session_date", DateType()),
    StructField("venue",      StringType()),
    StructField("sport",      StringType()),
    StructField("fps",        DoubleType()),
])

create("silver", "players", [
    StructField("player_id",  StringType()),
    StructField("name",       StringType()),
    StructField("age_group",  StringType()),
    StructField("team",       StringType()),
])

create("silver", "session_players", [
    StructField("session_id", StringType()),
    StructField("track_id",   IntegerType()),
    StructField("player_id",  StringType()),
])

create("silver", "tracks", [
    StructField("session_id",  StringType()),
    StructField("frame",       IntegerType()),
    StructField("track_id",    IntegerType()),
    StructField("timestamp_s", DoubleType()),
    StructField("court_x_m",   DoubleType()),
    StructField("court_y_m",   DoubleType()),
])

# --- GOLD (report-ready) ---
create("gold", "player_session_stats", [
    StructField("session_id",     StringType()),
    StructField("session_date",   DateType()),
    StructField("player_id",      StringType()),
    StructField("player_name",    StringType()),
    StructField("distance_m",     DoubleType()),
    StructField("avg_speed_kmh",  DoubleType()),
    StructField("max_speed_kmh",  DoubleType()),
    StructField("active_seconds", DoubleType()),
])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import numpy as np, pandas as pd
from datetime import date
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType, DateType
)

rng = np.random.default_rng(42)
session_id = "S001"
fps = 25
duration_s = 20 * 60
n_frames = fps * duration_s
players = [("P1","Ali"),("P2","Zeynep"),("P3","Mert"),("P4","Elif"),("P5","Kaan")]

# --- Generate tracks ---
rows = []
for tid, (pid, _name) in enumerate(players, start=1):
    x, y = rng.uniform(2, 26), rng.uniform(2, 13)
    for f in range(n_frames):
        x += rng.normal(0, 0.10); y += rng.normal(0, 0.10)
        x = min(28, max(0, x));   y = min(15, max(0, y))
        rows.append((session_id, f, tid, f/fps, float(x), float(y)))

tracks_pdf = pd.DataFrame(rows, columns=["session_id","frame","track_id","timestamp_s","court_x_m","court_y_m"])

tracks_schema = StructType([
    StructField("session_id",  StringType()),
    StructField("frame",       IntegerType()),
    StructField("track_id",    IntegerType()),
    StructField("timestamp_s", DoubleType()),
    StructField("court_x_m",   DoubleType()),
    StructField("court_y_m",   DoubleType()),
])
spark.createDataFrame(tracks_pdf, schema=tracks_schema) \
    .write.mode("append").saveAsTable("silver.tracks")

# --- Sessions ---
sessions_pdf = pd.DataFrame(
    [(session_id, date.today(), "Pilot Hall", "basketball", float(fps))],
    columns=["session_id","session_date","venue","sport","fps"])

sessions_schema = StructType([
    StructField("session_id",   StringType()),
    StructField("session_date", DateType()),
    StructField("venue",        StringType()),
    StructField("sport",        StringType()),
    StructField("fps",          DoubleType()),
])
spark.createDataFrame(sessions_pdf, schema=sessions_schema) \
    .write.mode("append").saveAsTable("silver.sessions")

# --- Players ---
players_pdf = pd.DataFrame(players, columns=["player_id","name"]).assign(age_group="U12", team="A")
spark.createDataFrame(players_pdf).write.mode("append").saveAsTable("silver.players")

# --- Session-player mapping ---
sp_pdf = pd.DataFrame(
    [(session_id, tid, pid) for tid,(pid,_) in enumerate(players, start=1)],
    columns=["session_id","track_id","player_id"])

sp_schema = StructType([
    StructField("session_id", StringType()),
    StructField("track_id",   IntegerType()),
    StructField("player_id",  StringType()),
])
spark.createDataFrame(sp_pdf, schema=sp_schema) \
    .write.mode("append").saveAsTable("silver.session_players")

print("synthetic session loaded")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql import functions as F, Window

w = Window.partitionBy("session_id","track_id").orderBy("frame")

tracks = spark.table("silver.tracks")
sessions = spark.table("silver.sessions").select("session_id","session_date","fps")
sp = spark.table("silver.session_players")
pl = spark.table("silver.players").select("player_id", F.col("name").alias("player_name"))

step = (tracks
    .withColumn("dx", F.col("court_x_m") - F.lag("court_x_m").over(w))
    .withColumn("dy", F.col("court_y_m") - F.lag("court_y_m").over(w))
    .withColumn("dt", F.col("timestamp_s") - F.lag("timestamp_s").over(w))
    .withColumn("step_m", F.sqrt(F.col("dx")**2 + F.col("dy")**2))
    .filter(F.col("step_m") > 0.05)                         # jitter filter
    .withColumn("speed_kmh", (F.col("step_m")/F.col("dt"))*3.6))

per_track = (step.groupBy("session_id","track_id")
    .agg(F.sum("step_m").alias("distance_m"),
         F.avg("speed_kmh").alias("avg_speed_kmh"),
         F.max("speed_kmh").alias("max_speed_kmh"),
         F.sum("dt").alias("active_seconds")))

gold = (per_track
    .join(sp, ["session_id","track_id"])
    .join(pl, "player_id")
    .join(sessions.select("session_id","session_date"), "session_id")
    .select("session_id","session_date","player_id","player_name",
            "distance_m","avg_speed_kmh","max_speed_kmh","active_seconds"))

gold.write.mode("overwrite").saveAsTable("gold.player_session_stats")

display(spark.table("gold.player_session_stats"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
