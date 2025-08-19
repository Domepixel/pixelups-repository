# supabase.py

from supabase import create_client, Client
import os

# Sustituye con tus propias claves
SUPABASE_URL = "https://dcibcbiezuhbjrgipxfi.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRjaWJjYmllenVoYmpyZ2lweGZpIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NDQ2MzkyMCwiZXhwIjoyMDcwMDM5OTIwfQ.jqYwhTC9hCfYbSuAu3eoFZ4CwyURRPJwjlFfYBWPOAM"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
