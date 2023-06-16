#include "AES.h"
#include "tabs/AES_cpu.tab"
#include <assert.h>
#include <cstring>

#define DIR_NONE 0
#define DIR_ENCRYPT 1
#define DIR_DECRYPT 2
#define DIR_BOTH (DIR_ENCRYPT | DIR_DECRYPT)

#define FULL_UNROLL

#ifdef _MSC_VER
#define SWAP(x) (_lrotl(x, 8) & 0x00ff00ff | _lrotr(x, 8) & 0xff00ff00)
#define GETWORD(p) SWAP(*((uint *)(p)))
#define PUTWORD(ct, st) (*((uint *)(ct)) = SWAP((st)))
#else
#define GETWORD(pt) (((uint)(pt)[0] << 24) ^ ((uint)(pt)[1] << 16) ^ ((uint)(pt)[2] <<  8) ^ ((uint)(pt)[3]))
#define PUTWORD(ct, st) ((ct)[0] = (uchar)((st) >> 24), (ct)[1] = (uchar)((st) >> 16), (ct)[2] = (uchar)((st) >>  8), (ct)[3] = (uchar)(st), (st))
#endif

void ExpandKey(const uchar *cipherKey, uint keyBits, uint *e_sched, int Nr) {
    uint *rek = e_sched;
    uint i = 0;
    uint temp;
    rek[0] = GETWORD(cipherKey     );
    rek[1] = GETWORD(cipherKey +  4);
    rek[2] = GETWORD(cipherKey +  8);
    rek[3] = GETWORD(cipherKey + 12);
    if (keyBits == 128) {
        for (;;) {
            temp  = rek[3];
            rek[4] = rek[0] ^
                (Te4[(temp >> 16) & 0xff] & 0xff000000) ^
                (Te4[(temp >>  8) & 0xff] & 0x00ff0000) ^
                (Te4[(temp      ) & 0xff] & 0x0000ff00) ^
                (Te4[(temp >> 24)       ] & 0x000000ff) ^
                rcon[i];
            rek[5] = rek[1] ^ rek[4];
            rek[6] = rek[2] ^ rek[5];
            rek[7] = rek[3] ^ rek[6];
            if (++i == 10) {
                Nr = 10;
                return;
            }
            rek += 4;
        }
    }
    rek[4] = GETWORD(cipherKey + 16);
    rek[5] = GETWORD(cipherKey + 20);
    if (keyBits == 192) {
        for (;;) {
            temp = rek[ 5];
            rek[ 6] = rek[ 0] ^
                (Te4[(temp >> 16) & 0xff] & 0xff000000) ^
                (Te4[(temp >>  8) & 0xff] & 0x00ff0000) ^
                (Te4[(temp      ) & 0xff] & 0x0000ff00) ^
                (Te4[(temp >> 24)       ] & 0x000000ff) ^
                rcon[i];
            rek[ 7] = rek[ 1] ^ rek[ 6];
            rek[ 8] = rek[ 2] ^ rek[ 7];
            rek[ 9] = rek[ 3] ^ rek[ 8];
            if (++i == 8) {
                Nr = 12;
                return;
            }
            rek[10] = rek[ 4] ^ rek[ 9];
            rek[11] = rek[ 5] ^ rek[10];
            rek += 6;
        }
    }
    rek[6] = GETWORD(cipherKey + 24);
    rek[7] = GETWORD(cipherKey + 28);
    if (keyBits == 256) {
        for (;;) {
            temp = rek[ 7];
            rek[ 8] = rek[ 0] ^
                (Te4[(temp >> 16) & 0xff] & 0xff000000) ^
                (Te4[(temp >>  8) & 0xff] & 0x00ff0000) ^
                (Te4[(temp      ) & 0xff] & 0x0000ff00) ^
                (Te4[(temp >> 24)       ] & 0x000000ff) ^
                rcon[i];
            rek[ 9] = rek[ 1] ^ rek[ 8];
            rek[10] = rek[ 2] ^ rek[ 9];
            rek[11] = rek[ 3] ^ rek[10];
            if (++i == 7) {
                Nr = 14;
                return;
            }
            temp = rek[11];
            rek[12] = rek[ 4] ^
                (Te4[(temp >> 24)       ] & 0xff000000) ^
                (Te4[(temp >> 16) & 0xff] & 0x00ff0000) ^
                (Te4[(temp >>  8) & 0xff] & 0x0000ff00) ^
                (Te4[(temp      ) & 0xff] & 0x000000ff);
            rek[13] = rek[ 5] ^ rek[12];
            rek[14] = rek[ 6] ^ rek[13];
            rek[15] = rek[ 7] ^ rek[14];
            rek += 8;
        }
    }
    Nr = 0; // this should never happen
}

void InvertKey(uint *e_sched, uint *d_sched, int Nr) {
    uint *rek = e_sched;
    uint *rdk = d_sched;
    assert(Nr == 10 || Nr == 12 || Nr == 14);
    rek += 4*Nr;
    /* apply the inverse MixColumn transform to all round keys but the first and the last: */
    memcpy(rdk, rek, 16);
    rdk += 4;
    rek -= 4;
    for (uint r = 1; r < Nr; r++) {
        rdk[0] =
            Td0[Te4[(rek[0] >> 24)       ] & 0xff] ^
            Td1[Te4[(rek[0] >> 16) & 0xff] & 0xff] ^
            Td2[Te4[(rek[0] >>  8) & 0xff] & 0xff] ^
            Td3[Te4[(rek[0]      ) & 0xff] & 0xff];
        rdk[1] =
            Td0[Te4[(rek[1] >> 24)       ] & 0xff] ^
            Td1[Te4[(rek[1] >> 16) & 0xff] & 0xff] ^
            Td2[Te4[(rek[1] >>  8) & 0xff] & 0xff] ^
            Td3[Te4[(rek[1]      ) & 0xff] & 0xff];
        rdk[2] =
            Td0[Te4[(rek[2] >> 24)       ] & 0xff] ^
            Td1[Te4[(rek[2] >> 16) & 0xff] & 0xff] ^
            Td2[Te4[(rek[2] >>  8) & 0xff] & 0xff] ^
            Td3[Te4[(rek[2]      ) & 0xff] & 0xff];
        rdk[3] =
            Td0[Te4[(rek[3] >> 24)       ] & 0xff] ^
            Td1[Te4[(rek[3] >> 16) & 0xff] & 0xff] ^
            Td2[Te4[(rek[3] >>  8) & 0xff] & 0xff] ^
            Td3[Te4[(rek[3]      ) & 0xff] & 0xff];
        rdk += 4;
        rek -= 4;
    }
    memcpy(rdk, rek, 16);
}

void makeKey(const uchar *cipherKey, uint keySize, uint dir, uint *e_sched,uint *d_sched, int Nr) {
    switch (keySize) {
    case 16:
    case 24:
    case 32:
        keySize <<= 3; // key size is now in bits
        break;
    case 128:
    case 192:
    case 256:
        break;
    default:
        throw "Invalid AES key size";
    }
    // assert(dir >= DIR_NONE && dir <= DIR_BOTH);
    assert(dir <= DIR_BOTH);
    if (dir != DIR_NONE) {
        ExpandKey(cipherKey, keySize, e_sched, Nr);

	    //printHexArray(e_sched, 44);
//        checkCudaErrors(cudaMemcpy(ce_sched, e_sched, sizeof(e_sched), cudaMemcpyHostToDevice));
        if (dir & DIR_DECRYPT) {
            InvertKey(e_sched, d_sched, Nr);
//            checkCudaErrors(cudaMemcpy(cd_sched, d_sched, sizeof(e_sched), cudaMemcpyHostToDevice));
        }
    }
}
