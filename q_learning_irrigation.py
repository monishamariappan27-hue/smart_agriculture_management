import numpy as np
import random
import joblib

# States
states = ["Dry", "Optimal", "Wet"]

# Actions
actions = ["Increase Water", "Maintain", "Decrease Water"]

# Q Table
Q = np.zeros((len(states), len(actions)))

alpha = 0.1
gamma = 0.9
epsilon = 0.1

rewards = {
    "Dry": [-1, -2, 5],
    "Optimal": [1, 10, 1],
    "Wet": [5, -2, -1]
}

# Training
for episode in range(1000):

    state = random.randint(0, 2)

    if random.uniform(0, 1) < epsilon:
        action = random.randint(0, 2)
    else:
        action = np.argmax(Q[state])

    reward = rewards[states[state]][action]

    next_state = random.randint(0, 2)

    Q[state, action] = Q[state, action] + alpha * (
        reward + gamma * np.max(Q[next_state]) - Q[state, action]
    )

print("\nQ Table:\n")
print(Q)

# Save Q Table
joblib.dump(
    Q,
    r"C:\Users\monis\OneDrive\Desktop\Smart_Agriculture_System\models\q_table.pkl"
)

print("\nQ Table saved successfully!")