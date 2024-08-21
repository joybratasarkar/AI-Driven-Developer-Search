document.getElementById('search-form').addEventListener('submit', async (event) => {
    event.preventDefault();

    const query = document.getElementById('query').value;

    try {
        const response = await fetch('http://127.0.0.1:8000/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query: query })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        displayResults(result);
    } catch (error) {
        console.error('Error during search:', error);
    }
});

function displayResults(result) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = ''; // Clear previous results

    if (result.top_profiles) {
        result.top_profiles.forEach(profile => {
            const initials = getInitials(profile.name);

            const profileCard = `
                <div class="profile-card">
                    <div class="profile-initials">${initials}</div>
                    <div class="profile-info">
                        <h2>${profile.name}</h2>
                        <p>Skills: ${profile.skills.join(', ')}</p>
                        <p>Experience: ${profile.experience} years</p>
                        <p>Location: ${profile.location}</p>
                        <p>Job Title: ${profile.job_title}</p>
                        <p>Cosine Similarity Score: ${profile.cosine_similarity_score}</p>
                        <div class="skills">
                            ${profile.skills.map(skill => `<span>${skill}</span>`).join('')}
                        </div>
                    </div>
                    <div class="actions">
                        <button class="btn">Show Interest</button>
                        <button class="btn archive">Archive</button>
                    </div>
                </div>
            `;
            resultsDiv.innerHTML += profileCard;
        });
    } else if (result.clarification_needed) {
        resultsDiv.innerHTML = `<p>${result.clarification_needed}</p>`;
    } else {
        resultsDiv.innerHTML = '<p>No results found.</p>';
    }
}

function getInitials(name) {
    const nameParts = name.split(' ');
    const firstInitial = nameParts[0].charAt(0).toUpperCase();
    const lastInitial = nameParts[nameParts.length - 1].charAt(0).toUpperCase();
    return `${firstInitial}${lastInitial}`;
}
