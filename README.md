# Inspiration
Who doesn't love grid games ? We do atleast, and during the hackathon we kept thinking about making a bot that feels stubborn and a little reckless but still smart. RevRage came from watching classic tron duels and wanting an agent that looks calm then suddenly takes space like crazy when it sees the opening.

# What it does
RevRage plays Tron on a 20 by 18 board. It tries to claim territory fast with a mix of voronoi control and flood fill awareness, then uses boosts only when they clearly give a jump in safety or a catch up on space. It avoids obvious traps and tries to cut the opponent off without throwing itself into a wall.

# How we built it
We started with a basic reachable area counter and a simple distance measure. Then we added voronoi scoring to estimate who owns which cells first. After that we plugged in a lightweight flood fill to compare near future space. Finally we added a careful boost trigger that checks two steps ahead and only fires when the landing is safe and the gain is significant. Most of this was done in one long night with a lot of coffee and some messy notebooks.

# Challenges we ran into
Timing out during move decisions was a pain. We also ran into some weird wrap around bugs because the board is a torus and sometimes we forgot a mod operation. Balancing aggression versus safety was tricky too. The bot sometimes tunnel visioned and we had to tune the weights so it does not throw a lead.

# Accomplishments that we're proud of
RevRage learned to wait patiently then explode forward when the board geometry says go. The boost usage feels surprisingly human. We are also happy that the agent stays responsive and passes the local tester without slowing down. We also trained it against our own rl based model and seeing RevRage adapt and hold its ground was very cool.

# What we learned
Voronoi is a strong but not perfect signal for tron. It needs help from local degree and flood fill to avoid fake space. We also learned that small constant time checks can save many late game blunders, like verifying degree after a two step boost. Training against our rl model taught us where our heuristics were weak and how to patch the gaps without slowing the bot.

# What's next for RevRage
We want to add a tiny depth search just for tight corridors, improve cut detection with a better articulation check, and maybe train some weights from more self play including our rl model as a sparring partner. Also cleaner logging and nicer visual dumps would make debugging way easier.
