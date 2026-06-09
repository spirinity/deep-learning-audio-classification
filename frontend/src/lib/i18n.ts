export type Language = "en" | "id";

export const COPY = {
  en: {
    language: "Language",
    headerTitle: "Sound Classifier Comparison",
    headerPrefix: "Compare",
    headerMiddle: "and",
    headerSuffix: "on the 50-class ESC-50 sound dataset.",
    uploadLabel: "Upload audio",
    uploadTitle: "Start by uploading your sound file",
    uploadMeta: "WAV or MP3, up to",
    dropText: "Drag & drop audio here, or",
    browse: "browse",
    uploadHint: "max",
    minDuration: "min 1 second",
    unsupportedType: "Unsupported file type. Use .wav or .mp3.",
    fileTooLarge: "File too large",
    maximumIs: "Maximum is",
    clear: "Clear",
    compareMethods: "Compare Methods",
    comparing: "Comparing...",
    backendNotReady: "Backend model is not ready yet.",
    backendRetry: "Retrying automatically...",
    modelLoadingTitle: "Loading the LAION-CLAP model...",
    modelLoadingDescription:
      "First start takes ~30 seconds while the model loads into memory.",
    methodsWork: "How the methods work",
    metricsTitle: "ESC-50 evaluation metrics",
    topPrediction: "Top prediction per method",
    logregUnavailable: "Logistic Regression is unavailable.",
    logregEnable:
      "Run python evaluate_methods.py and restart the backend to enable it.",
    processedIn: "Processed in",
    processedSuffix:
      "CLAP audio embedding is computed once, then shared across methods.",
    confidenceComparison: "Top-1 confidence comparison",
    topRankings: "Top-10 rankings by method",
    footer:
      "Based on \"A Multimodal Prototypical Approach for Unsupervised Sound Classification\" · INTERSPEECH 2023",
    setupLoadFailed: "The model failed to load on the backend.",
    setupFilesMissing: "Some required files are missing on the backend.",
    optional: "optional",
    setupInstruction:
      "Start the backend with uvicorn api:app --port 8000 and ensure the LAION-CLAP model is downloaded.",
    ready: "Ready",
    unavailable: "Unavailable",
    methodDescriptions: {
      "Zero-Shot CLAP":
        "The audio embedding is compared directly against ESC-50 label text embeddings.",
      "Proto-LC":
        "The audio embedding is compared with paper-inspired ESC-50 class prototypes.",
      "Logistic Regression":
        "A supervised classifier trained on top of LAION-CLAP audio embeddings.",
    },
    noMetrics:
      "No evaluation results yet. Run python evaluate_methods.py to generate comparison_metrics.csv.",
    metricMethod: "Method",
    metricAccuracy: "Accuracy",
    metricMacroF1: "Macro F1",
    metricTop3: "Top-3 Acc",
    metricInference: "Avg. inference",
    metricsFootnote:
      "Mean across ESC-50 5-fold cross-validation. Inference time is per sample on CLAP embeddings (model encoding excluded).",
    label: "Label",
    score: "Score",
    raw: "Raw",
    category: "Category",
  },
  id: {
    language: "Bahasa",
    headerTitle: "Perbandingan Klasifikasi Suara",
    headerPrefix: "Bandingkan",
    headerMiddle: "dan",
    headerSuffix: "pada dataset suara ESC-50 dengan 50 kelas.",
    uploadLabel: "Unggah audio",
    uploadTitle: "Mulai dengan mengunggah file suara",
    uploadMeta: "WAV atau MP3, hingga",
    dropText: "Tarik dan lepas audio di sini, atau",
    browse: "pilih file",
    uploadHint: "maks",
    minDuration: "min 1 detik",
    unsupportedType: "Tipe file tidak didukung. Gunakan .wav atau .mp3.",
    fileTooLarge: "File terlalu besar",
    maximumIs: "Maksimum",
    clear: "Hapus",
    compareMethods: "Bandingkan Metode",
    comparing: "Membandingkan...",
    backendNotReady: "Model backend belum siap.",
    backendRetry: "Mencoba ulang otomatis...",
    modelLoadingTitle: "Memuat model LAION-CLAP...",
    modelLoadingDescription:
      "Start pertama membutuhkan sekitar 30 detik saat model dimuat ke memori.",
    methodsWork: "Cara kerja metode",
    metricsTitle: "Metrik evaluasi ESC-50",
    topPrediction: "Prediksi teratas per metode",
    logregUnavailable: "Logistic Regression belum tersedia.",
    logregEnable:
      "Jalankan python evaluate_methods.py lalu restart backend untuk mengaktifkannya.",
    processedIn: "Diproses dalam",
    processedSuffix:
      "Embedding audio CLAP dihitung sekali, lalu dibagikan ke semua metode.",
    confidenceComparison: "Perbandingan confidence Top-1",
    topRankings: "Peringkat Top-10 per metode",
    footer:
      "Berdasarkan \"A Multimodal Prototypical Approach for Unsupervised Sound Classification\" · INTERSPEECH 2023",
    setupLoadFailed: "Model gagal dimuat di backend.",
    setupFilesMissing: "Beberapa file wajib belum tersedia di backend.",
    optional: "opsional",
    setupInstruction:
      "Jalankan backend dengan uvicorn api:app --port 8000 dan pastikan model LAION-CLAP sudah diunduh.",
    ready: "Siap",
    unavailable: "Tidak tersedia",
    methodDescriptions: {
      "Zero-Shot CLAP":
        "Audio embedding dibandingkan langsung dengan text embedding label ESC-50.",
      "Proto-LC":
        "Audio embedding dibandingkan dengan prototype kelas ESC-50 berbasis paper.",
      "Logistic Regression":
        "Classifier supervised yang dilatih di atas embedding audio LAION-CLAP.",
    },
    noMetrics:
      "Belum ada hasil evaluasi. Jalankan python evaluate_methods.py untuk membuat comparison_metrics.csv.",
    metricMethod: "Metode",
    metricAccuracy: "Akurasi",
    metricMacroF1: "Macro F1",
    metricTop3: "Akurasi Top-3",
    metricInference: "Inferensi rata-rata",
    metricsFootnote:
      "Rata-rata dari cross-validation 5-fold ESC-50. Waktu inferensi per sampel pada embedding CLAP (encoding model tidak dihitung).",
    label: "Label",
    score: "Skor",
    raw: "Raw",
    category: "Kategori",
  },
} as const;

export type Copy = (typeof COPY)[Language];
