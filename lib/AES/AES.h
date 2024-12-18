/**
 * AES.h
 *
 * The Advanced Encryption Standard (AES, aka AES) block cipher,
 * designed by J. Daemen and V. Rijmen.
 *
 * @author Paulo S. L. M. Barreto
 *
 * This software is hereby placed in the public domain.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef __AES_H
#define __AES_H

#include "BlockCipher.h"

#ifndef USUAL_TYPES
#define USUAL_TYPES
typedef unsigned char   uchar;
typedef unsigned int    uint;   /* assuming sizeof(uint) == 4 */
#endif /* USUAL_TYPES */

#ifndef AES_BLOCKBITS
#define AES_BLOCKBITS   128
#endif
#if AES_BLOCKBITS != 128
#error "AES_BLOCKBITS must be 128"
#endif

#ifndef AES_BLOCKSIZE
#define AES_BLOCKSIZE   16 /* uchars */
#endif
#if AES_BLOCKSIZE != 16
#error "AES_BLOCKSIZE must be 16"
#endif

#ifndef AES_MINKEYBITS
#define AES_MINKEYBITS  128
#endif
#if AES_MINKEYBITS != 128
#error "AES_MINKEYBITS must be 128"
#endif

#ifndef AES_MINKEYSIZE
#define AES_MINKEYSIZE  16 /* uchars */
#endif
#if AES_MINKEYSIZE != 16
#error "AES_MINKEYSIZE must be 16"
#endif

#ifndef AES_MAXKEYBITS
#define AES_MAXKEYBITS  256
#endif
#if AES_MAXKEYBITS != 256
#error "AES_MAXKEYBITS must be 256"
#endif

#ifndef AES_MAXKEYSIZE
#define AES_MAXKEYSIZE  32 /* uchars */
#endif
#if AES_MAXKEYSIZE != 32
#error "AES_MAXKEYSIZE must be 32"
#endif

#define MAXKC   (AES_MAXKEYBITS/32)
#define MAXKB   (AES_MAXKEYBITS/8)
#define MAXNR   14


#ifndef TTABLE
#define TTABLE 1024
#endif

class AES: public BlockCipher {

public:

    AES();
    virtual ~AES();

    /**
     * Block size in bits.
     */
    inline uint blockBits() const {
        return AES_BLOCKBITS;
    }

    /**
     * Block size in uchars.
     */
    inline uint blockSize() const {
        return AES_BLOCKSIZE;
    }

    /**
     * Key size in bits.
     */
    inline uint keyBits() const {
        return (Nr - 6) << 5;
    }

    /**
     * Key size in uchars.
     */
    inline uint keySize() const {
        return (Nr - 6) << 2;
    }

    /**
     * Convert one data block from uchar[] to uint[] representation.
     */
    void uchar2int(const uchar *b, uint *i);

    /**
     * Convert one data block from int[] to uchar[] representation.
     */
    void int2uchar(const uint *i, uchar *b);

    void makeKey(const uchar *cipherKey, uint keyBits, uint dir);

    void encrypt(const uint *pt, uint *ct);

    void decrypt(const uint *ct, uint *pt);

    void encrypt_ecb(const uint *pt, uint *ct, uint n);

    void encrypt_ecb_async(const uint *pt, uint *ct, uint n);

    uint Nr;

    // Pointers to GPU key schedules
    uint *ce_sched;
    uint *cd_sched;

private:

    // static void Initialize();
    void ExpandKey(const uchar *cipherKey, uint keyBits);
    void InvertKey();
    uint e_sched[4*(MAXNR + 1)];
    uint d_sched[4*(MAXNR + 1)];

};

#endif /* __AES_H */
