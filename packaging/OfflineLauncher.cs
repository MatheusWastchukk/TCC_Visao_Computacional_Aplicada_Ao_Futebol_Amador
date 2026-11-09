using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

[assembly: AssemblyVersion("@APP_VERSION@.0")]
[assembly: AssemblyProduct("Goal Event Review")]
[assembly: AssemblyTitle("Goal Event Review — pacote offline")]

class Bundle {
    public long Offset, Length, Expanded;
    public byte[] Hash;
    public string Id, Root, Target, Self;
    public static Bundle Read(string rootOverride) {
        var b = new Bundle();
        b.Self = Assembly.GetExecutingAssembly().Location;
        using (var stream = File.OpenRead(b.Self)) {
            if (stream.Length < 64) throw new IOException("O pacote está incompleto.");
            stream.Seek(-64, SeekOrigin.End);
            using (var reader = new BinaryReader(stream)) {
                b.Offset = reader.ReadInt64(); b.Length = reader.ReadInt64(); b.Expanded = reader.ReadInt64(); b.Hash = reader.ReadBytes(32);
                if (Encoding.ASCII.GetString(reader.ReadBytes(8)) != "GER7Z001" || b.Offset < 0 || b.Length < 0 || b.Expanded < 0 || b.Offset + b.Length != stream.Length - 64)
                    throw new IOException("O pacote está incompleto ou foi alterado.");
            }
        }
        b.Id = BitConverter.ToString(b.Hash).Replace("-", "").ToLowerInvariant();
        b.Root = Path.Combine(rootOverride == null ? Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData) : Path.GetFullPath(rootOverride), "GoalEventReview", "offline");
        b.Target = Path.Combine(b.Root, "app-" + b.Id.Substring(0, 16));
        return b;
    }
    public bool Ready() {
        string marker = Path.Combine(Target, "bundle.sha256");
        if (!File.Exists(marker) || File.ReadAllText(marker).Trim() != Id || !File.Exists(Path.Combine(Target, "Goal Event Review.exe"))) return false;
        foreach (string folder in new string[] { "backend", "engines/nvidia", "engines/amd" }) {
            string component = Path.Combine(Target, "resources", folder);
            if (!File.Exists(Path.Combine(component, "goal-review-api.exe")) || !Directory.Exists(Path.Combine(component, "_internal"))) return false;
        }
        return true;
    }
    public void Launch(string[] args) {
        var start = new ProcessStartInfo(Path.Combine(Target, "Goal Event Review.exe"));
        start.WorkingDirectory = Target; start.UseShellExecute = false;
        start.EnvironmentVariables.Remove("ELECTRON_RUN_AS_NODE");
        var text = new StringBuilder();
        foreach (string arg in args) { if (text.Length > 0) text.Append(' '); text.Append(Quote(arg)); }
        start.Arguments = text.ToString();
        Process.Start(start);
    }
    public static string Quote(string value) {
        var text = new StringBuilder("\""); int slashes = 0;
        foreach (char c in value) {
            if (c == '\\') { slashes++; continue; }
            if (c == '"') { text.Append('\\', slashes * 2 + 1); text.Append('"'); }
            else { text.Append('\\', slashes); text.Append(c); }
            slashes = 0;
        }
        text.Append('\\', slashes * 2); text.Append('"'); return text.ToString();
    }
    public string Safe(string path) {
        string full = Path.GetFullPath(path);
        if (!full.StartsWith(Path.GetFullPath(Root) + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
            throw new IOException("Caminho de preparação inválido.");
        return full;
    }
    public static string Extended(string path) {
        if (path.StartsWith("\\\\?\\")) return path;
        return path.StartsWith("\\\\") ? "\\\\?\\UNC\\" + path.Substring(2) : "\\\\?\\" + path;
    }
}

class SliceStream : Stream {
    readonly FileStream file; readonly long offset, length;
    public SliceStream(string path, long start, long size) { file = File.OpenRead(path); offset = start; length = size; file.Position = start; }
    public override bool CanRead { get { return true; } } public override bool CanWrite { get { return false; } }
    public override bool CanSeek { get { return true; } } public override long Length { get { return length; } }
    public override long Position { get { return file.Position - offset; } set { Seek(value, SeekOrigin.Begin); } }
    public override int Read(byte[] data, int start, int count) { return file.Read(data, start, (int)Math.Min(count, length - Position)); }
    public override long Seek(long distance, SeekOrigin origin) {
        long next = origin == SeekOrigin.Begin ? distance : origin == SeekOrigin.Current ? Position + distance : length + distance;
        if (next < 0 || next > length) throw new IOException("Posição inválida no pacote.");
        file.Position = offset + next; return next;
    }
    public override void Flush() { } public override void SetLength(long value) { throw new NotSupportedException(); }
    public override void Write(byte[] data, int start, int count) { throw new NotSupportedException(); }
    protected override void Dispose(bool disposing) { if (disposing) file.Dispose(); base.Dispose(disposing); }
}

class Preparation : Form {
    readonly Bundle bundle; readonly string[] arguments;
    readonly Label message = new Label(); readonly ProgressBar progress = new ProgressBar(); readonly Button cancel = new Button();
    volatile bool cancelled; bool finished;
    public Preparation(Bundle b, string[] args) {
        bundle = b; arguments = args;
        Text = "Goal Event Review — preparação inicial"; ClientSize = new Size(570, 185);
        StartPosition = FormStartPosition.CenterScreen; FormBorderStyle = FormBorderStyle.FixedDialog; MaximizeBox = false;
        Font = new Font("Segoe UI", 10); BackColor = Color.FromArgb(247, 245, 240);
        message.SetBounds(24, 20, 520, 68); message.Text = "Preparando os componentes CPU, NVIDIA e AMD…";
        progress.SetBounds(24, 98, 520, 18); progress.Maximum = 1000;
        cancel.SetBounds(434, 137, 110, 30); cancel.Text = "Cancelar";
        cancel.Click += delegate { cancelled = true; cancel.Enabled = false; };
        Controls.Add(message); Controls.Add(progress); Controls.Add(cancel);
        FormClosing += delegate(object sender, FormClosingEventArgs e) { if (!finished) { cancelled = true; cancel.Enabled = false; e.Cancel = true; } };
        Shown += delegate { Task.Run((Action)Prepare); };
    }
    void Check() { if (cancelled) throw new OperationCanceledException(); }
    void UpdateStatus(string text, double value) {
        if (IsDisposed) return;
        BeginInvoke((Action)delegate { message.Text = text; progress.Style = value < 0 ? ProgressBarStyle.Marquee : ProgressBarStyle.Blocks; if (value >= 0) progress.Value = Math.Max(0, Math.Min(1000, (int)(value * 1000))); });
    }
    void Prepare() {
        string staging = bundle.Safe(Path.Combine(bundle.Root, ".preparing-" + bundle.Id.Substring(0, 16)));
        try {
            Directory.CreateDirectory(bundle.Root);
            var buffer = new byte[1024 * 1024];
            using (var stream = new SliceStream(bundle.Self, bundle.Offset, bundle.Length))
            using (var sha = SHA256.Create()) {
                int read; long done = 0; var watch = Stopwatch.StartNew();
                while ((read = stream.Read(buffer, 0, buffer.Length)) > 0) {
                    Check(); sha.TransformBlock(buffer, 0, read, buffer, 0); done += read;
                    if (watch.ElapsedMilliseconds > 200) { UpdateStatus("Verificando o pacote offline…", 0.15 * done / bundle.Length); watch.Restart(); }
                }
                sha.TransformFinalBlock(new byte[0], 0, 0);
                if (BitConverter.ToString(sha.Hash) != BitConverter.ToString(bundle.Hash)) throw new IOException("A verificação do pacote falhou. Copie o executável novamente.");
            }
            ArchivePreparation.Extract(bundle, staging, Check, UpdateStatus);
            File.WriteAllText(Path.Combine(staging, "bundle.sha256"), bundle.Id);
            if (Directory.Exists(bundle.Target)) Directory.Delete(Bundle.Extended(bundle.Safe(bundle.Target)), true);
            Directory.Move(staging, bundle.Safe(bundle.Target));
            bundle.Launch(arguments);
        } catch (OperationCanceledException) {
        } catch (Exception error) {
            try { File.WriteAllText(Path.Combine(bundle.Root, "preparation-error.log"), error.ToString()); } catch { }
            Invoke((Action)delegate { MessageBox.Show(this, "Não foi possível preparar o aplicativo.\n\n" + error.Message, "Goal Event Review", MessageBoxButtons.OK, MessageBoxIcon.Error); });
        } finally {
            if (Directory.Exists(staging)) { try { Directory.Delete(Bundle.Extended(bundle.Safe(staging)), true); } catch { } }
            BeginInvoke((Action)delegate { finished = true; Close(); });
        }
    }
}

class OfflineLauncher {
    [STAThread]
    static void Main(string[] args) {
        AppContext.SetSwitch("Switch.System.IO.UseLegacyPathHandling", false);
        AppContext.SetSwitch("Switch.System.IO.BlockLongPaths", false);
        Application.EnableVisualStyles(); Application.SetCompatibleTextRenderingDefault(false);
        try {
            string rootOverride = null; var forward = new List<string>();
            foreach (string arg in args) {
                if (arg.StartsWith("--offline-root=")) rootOverride = arg.Substring("--offline-root=".Length);
                else forward.Add(arg);
            }
            string[] appArgs = forward.ToArray();
            Bundle bundle = Bundle.Read(rootOverride);
            if (bundle.Ready()) { bundle.Launch(appArgs); return; }
            using (var mutex = new Mutex(false, "Local\\GoalEventReviewOfflinePreparation")) {
                bool acquired;
                try { acquired = mutex.WaitOne(0); } catch (AbandonedMutexException) { acquired = true; }
                if (!acquired) { MessageBox.Show("A preparação do aplicativo já está em andamento.", "Goal Event Review"); return; }
                try { Application.Run(new Preparation(bundle, appArgs)); } finally { mutex.ReleaseMutex(); }
            }
        } catch (Exception error) { MessageBox.Show("Não foi possível abrir o pacote.\n\n" + error.Message, "Goal Event Review", MessageBoxButtons.OK, MessageBoxIcon.Error); }
    }
}
