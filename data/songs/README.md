# Song Files

Add your own songs to practice with Play-Along mode!

## Song Format

Each song is a JSON file with this structure:

```json
{
  "id": "song-id",
  "title": "Song Title",
  "artist": "Artist Name",
  "difficulty": "beginner/intermediate/advanced",
  "bpm": 120,
  "time_signature": "4/4",
  "count_in_bars": 1,
  "audio_file": "data/songs/audio/song.mp3",
  "timeline": [
    { "bar": 1, "beat": 1, "chord": "C" },
    { "bar": 3, "beat": 1, "chord": "G" }
  ]
}
```

## Adding Background Music

1. Get an MP3, WAV, or OGG file of the song
2. Place it in `data/songs/audio/` folder
3. Add `"audio_file": "data/songs/audio/yourfile.mp3"` to the song JSON

**Example:**
```json
{
  "id": "wonderwall",
  "title": "Wonderwall",
  "audio_file": "data/songs/audio/wonderwall.mp3",
  ...
}
```

## Controls During Play-Along

- **M** - Toggle music pause/play
- **+** - Increase volume
- **-** - Decrease volume
- **Q** - Quit

Default volume is 30% so you can hear your own playing!

## Tips

- Use instrumental/acoustic versions for better practice
- Lower BPM versions are easier to follow
- Make sure audio is synced to your chord timeline
