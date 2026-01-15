
export interface GitHubConfig {
  token: string;
  owner: string;
  repo: string;
  path: string; // default: 'public/data/model_result.json'
}

export const getGitHubConfig = (): GitHubConfig | null => {
  const stored = localStorage.getItem('GITHUB_CONFIG');
  return stored ? JSON.parse(stored) : null;
};

export const saveGitHubConfig = (config: GitHubConfig) => {
  localStorage.setItem('GITHUB_CONFIG', JSON.stringify(config));
};

/**
 * Encodes a string to Base64, handling UTF-8 characters (e.g., Chinese) correctly.
 */
const utf8_to_b64 = (str: string): string => {
  return window.btoa(unescape(encodeURIComponent(str)));
};

/**
 * Fetches the raw content of the file from GitHub.
 * Uses 'application/vnd.github.v3.raw' to get the raw file content directly,
 * avoiding Base64 decoding issues and the 1MB limit for the 'content' field.
 */
export const fetchFromGitHub = async (config: GitHubConfig) => {
  const { token, owner, repo, path } = config;
  // Cache bust to ensure we get the latest version
  const url = `https://api.github.com/repos/${owner}/${repo}/contents/${path}?t=${new Date().getTime()}`;
  
  const headers: any = {
    // CRITICAL: Request raw content to avoid JSON/Base64 parsing issues
    'Accept': 'application/vnd.github.v3.raw',
  };
  
  if (token) {
    headers['Authorization'] = `token ${token}`;
  }

  const res = await fetch(url, { headers });
  
  if (!res.ok) {
    if (res.status === 404) return null; // File doesn't exist
    if (res.status === 403) throw new Error("GitHub API rate limit exceeded or access denied. Check your token.");
    throw new Error(`GitHub Fetch Error: ${res.status} ${res.statusText}`);
  }
  
  const text = await res.text();
  
  // Guard against empty files causing JSON parse errors
  if (!text || text.trim() === '') {
    return null;
  }

  try {
    return JSON.parse(text);
  } catch (e) {
    console.error("Failed to parse GitHub response text:", text.substring(0, 100) + "...");
    throw new Error("Cloud file is not valid JSON.");
  }
};

/**
 * Uploads data to GitHub.
 * Handles reading the existing file (even if large) to merge data before saving.
 */
export const uploadToGitHub = async (
  config: GitHubConfig, 
  newContentObj: object, 
  message: string
) => {
  const { token, owner, repo, path } = config;
  const baseUrl = `https://api.github.com/repos/${owner}/${repo}/contents/${path}`;
  
  // Headers for metadata request (to get SHA)
  const metaHeaders: any = {
    'Accept': 'application/vnd.github.v3+json',
    'Authorization': `token ${token}`,
  };

  try {
    // 1. Get existing file metadata (SHA)
    // Add cache bust to metadata request too
    const metaUrl = `${baseUrl}?t=${new Date().getTime()}`;
    let sha: string | undefined;
    let existingContent: any = {};

    const getRes = await fetch(metaUrl, { headers: metaHeaders });
    
    if (getRes.status === 200) {
      const data = await getRes.json();
      sha = data.sha;
      
      // 2. Fetch actual content
      // STRICT CHECK: If we found the file (SHA exists), we MUST successfully read it 
      // before writing back. Otherwise we risk overwriting existing data (from other models) 
      // with an empty object if the fetch fails.
      
      const rawUrl = `${baseUrl}?t=${new Date().getTime()}`; // Cache bust raw content
      const rawRes = await fetch(rawUrl, {
        headers: {
          'Authorization': `token ${token}`,
          'Accept': 'application/vnd.github.v3.raw'
        }
      });
      
      if (!rawRes.ok) {
        // ABORT: Cannot read existing file
        throw new Error(`Critical: Remote file exists (SHA: ${sha}) but content download failed (${rawRes.status}). Aborting upload to prevent data loss.`);
      }

      const rawText = await rawRes.text();
      if (rawText && rawText.trim() !== '') {
        try {
          existingContent = JSON.parse(rawText);
          
          // Basic validation
          if (typeof existingContent !== 'object' || existingContent === null) {
             throw new Error("Remote content is not a JSON object");
          }
        } catch (e: any) {
          // ABORT: Cannot parse existing file
          console.error("Existing cloud file invalid JSON:", e);
          throw new Error(`Critical: Existing cloud file is invalid JSON. Aborting to prevent data corruption. Error: ${e.message}`);
        }
      }
    } else if (getRes.status === 404) {
      // File doesn't exist yet, safe to create new
      existingContent = {};
    } else if (getRes.status === 403) {
      throw new Error("GitHub API permission denied. Check your token scopes.");
    } else {
      throw new Error(`GitHub API Error (Get Metadata): ${getRes.statusText}`);
    }

    // 3. Merge new content with existing
    // This ensures we only update the key for the current model, preserving others
    const finalContent = {
      ...existingContent,
      ...newContentObj
    };

    // 4. Upload (PUT)
    // Note: The GitHub API has a ~100MB limit for raw uploads but via JSON/Base64 it's lower.
    // For this app's scale (text data), this is usually fine.
    const contentEncoded = utf8_to_b64(JSON.stringify(finalContent, null, 2));

    const putBody = {
      message,
      content: contentEncoded,
      sha, // Required if updating existing file
    };

    const putRes = await fetch(baseUrl, {
      method: 'PUT',
      headers: {
        'Authorization': `token ${token}`,
        'Accept': 'application/vnd.github.v3+json',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(putBody),
    });

    if (!putRes.ok) {
      const errData = await putRes.json();
      throw new Error(`GitHub API Error (Put): ${errData.message || putRes.statusText}`);
    }

    return await putRes.json();
  } catch (error: any) {
    throw new Error(`GitHub Sync Failed: ${error.message}`);
  }
};
