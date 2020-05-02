#ifndef PROFILER_H
#define PROFILER_H

#include <chrono>
#include <thread>
#include <fstream>
#include <algorithm>

struct ProfileResult {
    std::string name;
    long long start, end;
    uint32_t threadID;
};

struct InstrumentationSession {
    std::string name;  
    InstrumentationSession(const std::string &n) : name(n) {}
};

struct Instrumentor {
    InstrumentationSession *currentSession;
    std::ofstream outputStream;
    int profileCount;

    Instrumentor() : currentSession(nullptr), profileCount(0) {}
    
    void beginSession(const std::string& name, const std::string &filepath = "results.json") {
        outputStream.open(filepath);
        writeHeader();
        currentSession = new InstrumentationSession(name);
    }

    void endSession() {
        writeFooter();
        outputStream.close();
        delete currentSession;
        currentSession = nullptr;
        profileCount = 0;
    }

    void writeProfile(const ProfileResult& result) {
        if (profileCount++ > 0) outputStream << ",";

        std::string name = result.name;
        std::replace(name.begin(), name.end(), '"', '\'');

        outputStream << "{";
        outputStream << "\"cat\":\"function\",";
        outputStream << "\"dur\":" << (result.end - result.start) << ',';
        outputStream << "\"name\":\"" << name << "\",";
        outputStream << "\"ph\":\"X\",";
        outputStream << "\"pid\":0,";
        outputStream << "\"tid\":" << result.threadID << ",";
        outputStream << "\"ts\":" << result.start;
        outputStream << "}";

        outputStream.flush();
    }

    void writeHeader() {
        outputStream << "{\"otherData\": {}, \"traceEvents\":[";
        outputStream.flush();
    }

    void writeFooter() {
        outputStream << "]}";
        outputStream.flush();
    }

    static Instrumentor &get() {
        static Instrumentor instance;
        return instance;
    }
};

struct Timer {
    const char *name;
    std::chrono::time_point<std::chrono::high_resolution_clock> startTimepoint;
    bool stopped;

    Timer(const char *n) : name(n), stopped(false) {
        startTimepoint = std::chrono::high_resolution_clock::now();
    }

    void stop() {
        auto endTimepoint = std::chrono::high_resolution_clock::now();
        long long start = std::chrono::time_point_cast<std::chrono::milliseconds>(startTimepoint).time_since_epoch().count();
        long long end = std::chrono::time_point_cast<std::chrono::milliseconds>(endTimepoint).time_since_epoch().count();

        uint32_t threadID = std::hash<std::thread::id>{}(std::this_thread::get_id());
        Instrumentor::get().writeProfile({name, start, end, threadID});

        stopped = true;
    }

    ~Timer() {
        if (!stopped) {
            stop();
        }
    }
};

#endif