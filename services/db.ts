import { ModelType } from '../types';

const DB_NAME = 'CollectionVolumeDB';
const DB_VERSION = 1;
const STORE_MODELS = 'models';
const STORE_DATASETS = 'datasets';

// Open Database Connection
const openDB = (): Promise<IDBDatabase> => {
  return new Promise((resolve, reject) => {
    if (!window.indexedDB) {
      reject(new Error("Your browser does not support IndexedDB"));
      return;
    }
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);
    
    request.onupgradeneeded = (e) => {
      const db = (e.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains(STORE_MODELS)) {
        db.createObjectStore(STORE_MODELS); // Key: modelType string
      }
      if (!db.objectStoreNames.contains(STORE_DATASETS)) {
        db.createObjectStore(STORE_DATASETS); // Key: modelType string
      }
    };
  });
};

// Generic Transaction Helper
const performRequest = <T>(
  storeName: string, 
  mode: IDBTransactionMode, 
  operation: (store: IDBObjectStore) => IDBRequest<T>
): Promise<T> => {
  return new Promise(async (resolve, reject) => {
    try {
      const db = await openDB();
      const tx = db.transaction(storeName, mode);
      const store = tx.objectStore(storeName);
      const request = operation(store);
      
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    } catch (err) {
      reject(err);
    }
  });
};

const performDelete = (storeName: string, key: string): Promise<void> => {
    return new Promise(async (resolve, reject) => {
        try {
            const db = await openDB();
            const tx = db.transaction(storeName, 'readwrite');
            const store = tx.objectStore(storeName);
            const request = store.delete(key);
            
            tx.oncomplete = () => resolve();
            tx.onerror = () => reject(tx.error);
        } catch (err) {
            reject(err);
        }
    });
}

export const dbService = {
  saveModel: (key: string, data: any) => performRequest(STORE_MODELS, 'readwrite', store => store.put(data, key)),
  getModel: (key: string) => performRequest(STORE_MODELS, 'readonly', store => store.get(key)),
  
  saveData: (key: string, data: any) => performRequest(STORE_DATASETS, 'readwrite', store => store.put(data, key)),
  getData: (key: string) => performRequest(STORE_DATASETS, 'readonly', store => store.get(key)),
  
  clearAll: async (key: string) => {
     await Promise.all([
         performDelete(STORE_MODELS, key),
         performDelete(STORE_DATASETS, key)
     ]);
  }
};