#import <AppKit/AppKit.h>
#import <Foundation/Foundation.h>

@interface ProbeView : NSView
@end

@implementation ProbeView

- (BOOL)isOpaque {
    return NO;
}

- (void)drawRect:(NSRect)dirtyRect {
    [super drawRect:dirtyRect];
    [[NSColor colorWithRed:1 green:0 blue:0 alpha:0.9] setFill];
    NSRectFill(NSMakeRect(self.bounds.size.width - 140, self.bounds.size.height - 140, 80, 80));
}

@end

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        BOOL blocksMouse = NO;
        BOOL normalLevel = NO;
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--blocks-mouse") == 0) {
                blocksMouse = YES;
            } else if (strcmp(argv[i], "--normal-level") == 0) {
                normalLevel = YES;
            } else {
                fprintf(stderr, "Usage: mac_transparent_overlay [--blocks-mouse] [--normal-level]\n");
                return 2;
            }
        }

        NSApplication *application = [NSApplication sharedApplication];
        [application setActivationPolicy:NSApplicationActivationPolicyRegular];
        NSScreen *screen = NSScreen.mainScreen;
        if (screen == nil) {
            fprintf(stderr, "No main screen available.\n");
            return 1;
        }

        NSWindow *window = [[NSWindow alloc] initWithContentRect:screen.frame
                                                        styleMask:NSWindowStyleMaskBorderless
                                                          backing:NSBackingStoreBuffered
                                                            defer:NO];
        window.title = @"Transparent Overlay Probe";
        window.backgroundColor = NSColor.clearColor;
        window.opaque = NO;
        window.hasShadow = NO;
        window.ignoresMouseEvents = !blocksMouse;
        window.level = normalLevel ? NSNormalWindowLevel : NSStatusWindowLevel;
        window.collectionBehavior = NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorFullScreenAuxiliary;
        window.contentView = [[ProbeView alloc] initWithFrame:window.contentLayoutRect];
        [window orderFrontRegardless];

        printf(
            "Showing transparent overlay for 15 seconds; blocks mouse: %s; level: %s.\n",
            blocksMouse ? "yes" : "no",
            normalLevel ? "normal" : "status"
        );
        fflush(stdout);
        dispatch_after(
            dispatch_time(DISPATCH_TIME_NOW, 15 * NSEC_PER_SEC),
            dispatch_get_main_queue(),
            ^{
                [application terminate:nil];
            }
        );
        [application run];
    }

    return 0;
}
