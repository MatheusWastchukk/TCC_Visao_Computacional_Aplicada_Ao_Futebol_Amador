; Base CPU offline e componentes opcionais obtidos diretamente dos fornecedores.
[Setup]
AppId=br.edu.tcc.goaleventreview.online
AppName=Goal Event Review
AppVersion={#AppVersion}
AppPublisher=Goal Event Review
DefaultDirName={localappdata}\Programs\Goal Event Review Online
DefaultGroupName=Goal Event Review
PrivilegesRequired=lowest
SetupArchitecture=x64
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir={#OutputFolder}
OutputBaseFilename=GoalEventReview-{#AppVersion}-Online-Setup-x64
Compression=lzma2/normal
LZMADictionarySize=32768
LZMANumBlockThreads=1
SolidCompression=yes
WizardStyle=modern
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\Goal Event Review.exe
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "brazilianportuguese"; MessagesFile: "compiler:Languages\BrazilianPortuguese.isl"

[Tasks]
Name: "desktopicon"; Description: "Criar atalho na área de trabalho"; Flags: unchecked

[Files]
Source: "{#BaseSource}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#CatalogSource}"; DestDir: "{app}\resources"; DestName: "gpu_catalog.json"; Flags: ignoreversion; AfterInstall: InstallGpuRuntime

[Icons]
Name: "{group}\Goal Event Review"; Filename: "{app}\Goal Event Review.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\Goal Event Review"; Filename: "{app}\Goal Event Review.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\Goal Event Review.exe"; Description: "Iniciar Goal Event Review"; Flags: nowait postinstall skipifsilent

[Code]
var
  ModePage: TInputOptionWizardPage;
  DownloadPage: TDownloadWizardPage;

function SelectedMode: String;
begin
  case ModePage.SelectedValueIndex of
    1: Result := 'nvidia';
    2: Result := 'amd';
  else
    Result := 'cpu';
  end;
end;

procedure InitializeWizard;
var
  Mode: String;
begin
  ModePage := CreateInputOptionPage(wpSelectDir, 'Modo de processamento',
    'CPU incluída; GPU opcional com download durante a instalação',
    'Os componentes de GPU são software de terceiros, baixados por HTTPS das fontes oficiais PyTorch (download.pytorch.org / download-r2.pytorch.org) e AMD (repo.radeon.com).' + #13#10 +
    'Ao selecionar uma GPU, serão instaladas versões fixadas de PyTorch, torchvision e, para AMD, ROCm. Cada download é verificado por SHA-256. Os drivers de vídeo precisam estar instalados e não serão modificados.' + #13#10 +
    'O modo CPU funciona sem internet. Escolha GPU somente se houver uma placa compatível.', True, False);
  ModePage.Add('CPU — sem downloads adicionais');
  ModePage.Add('GPU NVIDIA — download de aproximadamente {#NvidiaDownloadMB} MB');
  ModePage.Add('GPU AMD — download de aproximadamente {#AmdDownloadMB} MB');
  Mode := Lowercase(ExpandConstant('{param:GPU|cpu}'));
  if Mode = 'nvidia' then ModePage.SelectedValueIndex := 1
  else if Mode = 'amd' then ModePage.SelectedValueIndex := 2
  else if Mode = 'cpu' then ModePage.SelectedValueIndex := 0
  else RaiseException('Opção inválida. Use /GPU=cpu, /GPU=nvidia ou /GPU=amd.');
  DownloadPage := CreateDownloadPage('Componentes oficiais de GPU',
    'Baixando e verificando os pacotes de terceiros selecionados...', nil);
end;

function HardwareAvailable(Mode: String): Boolean;
var
  Locator, Services, Adapters, Adapter: Variant;
  I: Integer;
  Description: String;
begin
  Result := False;
  Locator := CreateOleObject('WbemScripting.SWbemLocator');
  Services := Locator.ConnectServer('', 'root\CIMV2');
  Adapters := Services.ExecQuery('SELECT Name, PNPDeviceID FROM Win32_VideoController');
  for I := 0 to Adapters.Count - 1 do begin
    Adapter := Adapters.ItemIndex(I);
    Description := Uppercase(String(Adapter.Name) + ' ' + String(Adapter.PNPDeviceID));
    if ((Mode = 'nvidia') and ((Pos('VEN_10DE', Description) > 0) or (Pos('NVIDIA', Description) > 0))) or
       ((Mode = 'amd') and ((Pos('VEN_1002', Description) > 0) or (Pos('RADEON', Description) > 0))) then
      Result := True;
  end;
end;

procedure AddGpuDownloads;
begin
  #include DownloadEntries
end;

function NextButtonClick(CurPageID: Integer): Boolean;
var
  ErrorText: String;
begin
  Result := True;
  if (CurPageID <> wpReady) or (SelectedMode = 'cpu') then Exit;
  try
    if not HardwareAvailable(SelectedMode) then
      RaiseException('Nenhuma placa do fornecedor selecionado foi encontrada. Volte e escolha CPU ou a GPU disponível.');
    DownloadPage.Clear;
    AddGpuDownloads;
    DownloadPage.Show;
    try
      DownloadPage.Download;
    finally
      DownloadPage.Hide;
    end;
  except
    ErrorText := GetExceptionMessage;
    if not DownloadPage.AbortedByUser then
      SuppressibleMsgBox('Os componentes GPU não foram instalados. ' + ErrorText + #13#10 +
        'Você pode tentar novamente ou voltar e escolher CPU, que não exige downloads.', mbError, MB_OK, IDOK);
    Result := False;
  end;
end;

procedure InstallGpuRuntime;
var
  ExitCode: Integer;
begin
  if SelectedMode = 'cpu' then Exit;
  WizardForm.StatusLabel.Caption := 'Instalando os componentes oficiais de GPU...';
  if not ExecAndLogOutput(ExpandConstant('{app}\resources\backend\goal-review-api.exe'),
    '--install-runtime ' + SelectedMode + ' --runtime-downloads "' + ExpandConstant('{tmp}') +
    '" --gpu-root "' + ExpandConstant('{app}\resources\gpu') +
    '" --gpu-catalog "' + ExpandConstant('{app}\resources\gpu_catalog.json') + '"',
    '', SW_HIDE, ewWaitUntilTerminated, ExitCode, nil) then
    RaiseException('Não foi possível iniciar a instalação dos componentes GPU.');
  if ExitCode <> 0 then
    RaiseException('A instalação do componente GPU falhou. Execute novamente com CPU ou verifique os downloads. Código: ' + IntToStr(ExitCode));
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  ExitCode: Integer;
begin
  if CurUninstallStep <> usUninstall then Exit;
  if not ExecAndLogOutput(ExpandConstant('{app}\resources\backend\goal-review-api.exe'),
    '--remove-runtimes --gpu-root "' + ExpandConstant('{app}\resources\gpu') + '"',
    '', SW_HIDE, ewWaitUntilTerminated, ExitCode, nil) then
    RaiseException('Não foi possível remover os componentes GPU registrados.');
  if ExitCode <> 0 then RaiseException('Não foi possível remover os componentes GPU com segurança.');
end;
