#include "ARKitFaceReceiver.h"
#include "Async/Async.h"

/* ═══════════════════════════════════════════════════════════════════════════
   FARKitReceiverThread
   ═══════════════════════════════════════════════════════════════════════════ */

FARKitReceiverThread::FARKitReceiverThread(UARKitFaceReceiver* InOwner, FSocket* InSocket)
    : Owner(InOwner), Socket(InSocket)
{}

bool FARKitReceiverThread::Init()
{
    bRunning = true;
    return true;
}

uint32 FARKitReceiverThread::Run()
{
    TArray<uint8> Buffer;
    Buffer.SetNumUninitialized(2048);

    while (bRunning)
    {
        uint32 PendingSize = 0;
        if (!Socket->HasPendingData(PendingSize))
        {
            FPlatformProcess::Sleep(0.001f); // ~1 ms poll
            continue;
        }

        int32 BytesRead = 0;
        if (Socket->Recv(Buffer.GetData(), Buffer.Num(), BytesRead) && BytesRead > 0)
        {
            TArray<uint8> Packet(Buffer.GetData(), BytesRead);
            FARKitFaceFrame Frame;
            if (Owner->TryParsePacket(Packet, Frame))
            {
                Owner->EnqueueFrame(Frame);
            }
        }
    }
    return 0;
}

void FARKitReceiverThread::Stop()
{
    bRunning = false;
}

void FARKitReceiverThread::Exit() {}


/* ═══════════════════════════════════════════════════════════════════════════
   UARKitFaceReceiver
   ═══════════════════════════════════════════════════════════════════════════ */

UARKitFaceReceiver::UARKitFaceReceiver()
{
    PrimaryComponentTick.bCanEverTick = true;
    SmoothedFrame.Shapes.SetNumZeroed((int32)EARKitShape::COUNT);
}

// ── Lifecycle ────────────────────────────────────────────────────────────────

void UARKitFaceReceiver::BeginPlay()
{
    Super::BeginPlay();

    ISocketSubsystem* SocketSub = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM);
    if (!SocketSub)
    {
        UE_LOG(LogTemp, Error, TEXT("ARKitFaceReceiver: no socket subsystem"));
        return;
    }

    Socket = SocketSub->CreateSocket(NAME_DGram, TEXT("ARKitUDP"), false);
    if (!Socket)
    {
        UE_LOG(LogTemp, Error, TEXT("ARKitFaceReceiver: failed to create socket"));
        return;
    }

    Socket->SetNonBlocking(true);
    Socket->SetReuseAddr(true);

    TSharedRef<FInternetAddr> Addr = SocketSub->CreateInternetAddr();
    Addr->SetAnyAddress();
    Addr->SetPort(ListenPort);

    if (!Socket->Bind(*Addr))
    {
        UE_LOG(LogTemp, Error, TEXT("ARKitFaceReceiver: failed to bind port %d"), ListenPort);
        SocketSub->DestroySocket(Socket);
        Socket = nullptr;
        return;
    }

    RecvThread = new FARKitReceiverThread(this, Socket);
    Thread = FRunnableThread::Create(RecvThread, TEXT("ARKitRecv"),
                                     0, TPri_BelowNormal);

    UE_LOG(LogTemp, Log, TEXT("ARKitFaceReceiver: listening on UDP %d"), ListenPort);
}

void UARKitFaceReceiver::EndPlay(const EEndPlayReason::Type Reason)
{
    if (RecvThread) { RecvThread->Stop(); }
    if (Thread)     { Thread->WaitForCompletion(); delete Thread; Thread = nullptr; }
    if (RecvThread) { delete RecvThread; RecvThread = nullptr; }

    if (Socket)
    {
        ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->DestroySocket(Socket);
        Socket = nullptr;
    }

    Super::EndPlay(Reason);
}

void UARKitFaceReceiver::TickComponent(float DeltaTime, ELevelTick TickType,
                                        FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

    // Drain the pending frame (one per tick is fine at 60 fps; tracker runs at ~24)
    FARKitFaceFrame Pending;
    bool bGot = false;
    {
        FScopeLock Lock(&PendingLock);
        if (bHasPending)
        {
            Pending   = PendingFrame.GetValue();
            bHasPending = false;
            bGot = true;
        }
    }

    if (bGot)
    {
        ApplySmoothing(Pending);
        LatestFrame = SmoothedFrame;
        OnFaceFrameReceived.Broadcast(LatestFrame);
    }
}

// ── Thread-safe enqueue ──────────────────────────────────────────────────────

void UARKitFaceReceiver::EnqueueFrame(const FARKitFaceFrame& Frame)
{
    FScopeLock Lock(&PendingLock);
    PendingFrame = Frame;
    bHasPending  = true;
}

// ── Packet parser ────────────────────────────────────────────────────────────

bool UARKitFaceReceiver::TryParsePacket(const TArray<uint8>& Data, FARKitFaceFrame& OutFrame)
{
    if (Data.Num() < PACKET_BYTES)
    {
        return false;
    }

    const uint8* Ptr = Data.GetData();

    // Magic "ARKF"
    if (Ptr[0] != 'A' || Ptr[1] != 'R' || Ptr[2] != 'K' || Ptr[3] != 'F')
    {
        return false;
    }
    Ptr += 4;

    // Timestamp (double, 8 bytes, little-endian)
    double Timestamp;
    FMemory::Memcpy(&Timestamp, Ptr, 8); Ptr += 8;
    OutFrame.Timestamp = Timestamp;

    // Success flag
    OutFrame.bSuccess = (*Ptr != 0); Ptr += 1;

    // Helper — read one float LE
    auto ReadFloat = [&]() -> float {
        float V; FMemory::Memcpy(&V, Ptr, 4); Ptr += 4; return V;
    };

    // Quaternion (x y z w)
    float Qx = ReadFloat(), Qy = ReadFloat(), Qz = ReadFloat(), Qw = ReadFloat();
    OutFrame.HeadRotation = FQuat(Qx, -Qy, -Qz, Qw); // flip Y/Z for UE coord system

    // Euler (pitch yaw roll, degrees) — OSF gives OpenCV convention
    float EPitch = ReadFloat(), EYaw = ReadFloat(), ERoll = ReadFloat();
    OutFrame.HeadEuler = FRotator(-EPitch, EYaw, -ERoll); // remap to UE

    // Translation
    float Tx = ReadFloat(), Ty = ReadFloat(), Tz = ReadFloat();
    OutFrame.HeadTranslation = FVector(Tz, Tx, -Ty); // OpenCV→UE: Z fwd, X right, Y up

    // 61 blendshape floats
    OutFrame.Shapes.SetNumUninitialized((int32)EARKitShape::COUNT);
    for (int32 i = 0; i < (int32)EARKitShape::COUNT; ++i)
    {
        OutFrame.Shapes[i] = FMath::Clamp(ReadFloat(), 0.0f, 1.0f);
    }

    return true;
}

// ── Smoothing ────────────────────────────────────────────────────────────────

void UARKitFaceReceiver::ApplySmoothing(const FARKitFaceFrame& Raw)
{
    if (SmoothingAlpha <= 0.0f)
    {
        SmoothedFrame = Raw;
        return;
    }

    const float A = SmoothingAlpha;
    const float B = 1.0f - A;

    SmoothedFrame.Timestamp    = Raw.Timestamp;
    SmoothedFrame.bSuccess     = Raw.bSuccess;
    SmoothedFrame.HeadRotation = FQuat::Slerp(Raw.HeadRotation, SmoothedFrame.HeadRotation, A);
    SmoothedFrame.HeadEuler    = FMath::Lerp(Raw.HeadEuler.Vector(), SmoothedFrame.HeadEuler.Vector(), A).Rotation();
    SmoothedFrame.HeadTranslation = FMath::Lerp(Raw.HeadTranslation, SmoothedFrame.HeadTranslation, A);

    if (SmoothedFrame.Shapes.Num() != (int32)EARKitShape::COUNT)
    {
        SmoothedFrame.Shapes.SetNumZeroed((int32)EARKitShape::COUNT);
    }
    for (int32 i = 0; i < (int32)EARKitShape::COUNT; ++i)
    {
        SmoothedFrame.Shapes[i] = Raw.Shapes[i] * B + SmoothedFrame.Shapes[i] * A;
    }
}

// ── Blueprint helpers ─────────────────────────────────────────────────────────

float UARKitFaceReceiver::GetShape(EARKitShape Shape) const
{
    const int32 Idx = (int32)Shape;
    if (LatestFrame.Shapes.IsValidIndex(Idx))
    {
        return LatestFrame.Shapes[Idx];
    }
    return 0.0f;
}

TArray<float> UARKitFaceReceiver::GetAllShapes() const
{
    return LatestFrame.Shapes;
}
